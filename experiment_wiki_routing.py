#!/usr/bin/env python3
"""
For each Wikipedia title, log the routing decision at the LAST token of every MoE
layer of a trained checkpoint:

  per layer L, per title:
    expert_indices  : the top-K experts chosen at the last token (size K)
    expert_scores   : their routing scores (size K, after gating + L1 renorm)
    all_scores      : the full router score vector over all E experts
    w_gate_top10    : per chosen expert, the top-10 indices in w_gate where
                      <w_gate[e, j, :], h_last> is largest (size K x 10)

Padding question — short titles batched with longer ones: causal attention only
attends to past tokens, so right-padding does not affect the last *real* token's
hidden state. We just need each sample's true length-1 to index the last token.

Scope: this script targets the `default` (linear) router with sigmoid gating
(the deepseek variant). It rejects checkpoints with EMA z-score normalization
enabled — for those, the `all_scores` recomputation would diverge from the
scores the router used internally for selection.

Usage:
    uv run python experiment_wiki_routing.py \
        --run moe-1b-269440-deepseek \
        --step 39750 \
        --out runs/moe-1b-269440-deepseek/wiki_routing_step39750.npz \
        --max-titles 1000000

    # smoke test:
    uv run python experiment_wiki_routing.py --run moe-1b-269440-deepseek \
        --step 39750 --out /tmp/wiki_routing_smoke.npz --max-titles 2000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from olmo_core.data import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.moe import MoERouterGatingFunction, MoERouterType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig

REPO_DIR = Path(__file__).parent
RUNS_DIR = REPO_DIR / "runs"

TOP10 = 10  # how many w_gate hidden indices to keep per chosen expert


# ── model build / load ────────────────────────────────────────────────────────


def detect_router_config(run_dir: Path, step: int) -> dict:
    cfg = json.loads((run_dir / f"step{step}" / "config.json").read_text())
    return cfg["model"]["block"]["feed_forward_moe"]["router"]


def build_model() -> torch.nn.Module:
    """Mirror src/scripts/train/moe-1b.py architecture (deepseek variant)."""
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    d_model = 1024
    cfg = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=9,
        n_heads=8,
        num_experts=64,
        top_k=6,
        expert_hidden_size=d_model // 2,
        shared_expert_hidden_size=d_model,
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        # losses are training-only, irrelevant at inference:
        lb_loss_weight=None,
        z_loss_weight=None,
        init_std=0.02,
    )
    block = cfg.block
    assert isinstance(block, TransformerBlockConfig)
    moe = block.feed_forward_moe
    assert moe is not None
    moe.router.gating_function = MoERouterGatingFunction.sigmoid
    moe.router.bias_gamma = 1e-3
    moe.router.normalize_expert_weights = 1.0
    return cfg.build(init_device="cpu")


def load_model(run_dir: Path, step: int, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    rcfg = detect_router_config(run_dir, step)
    if rcfg["name"] != MoERouterType.default.value:
        raise SystemExit(
            f"This script targets the deepseek variant (router='default'); "
            f"got {rcfg['name']!r}. Generalize build_model() to support other variants."
        )
    if rcfg.get("ema_zscore_normalize", False):
        # `all_scores` recompute below assumes raw sigmoid(h @ Wᵀ); with z-score
        # normalisation the router computes sigmoid((logits-μ)/σ̂) using buffered
        # EMA stats, and those stats are NOT replayed here. Refusing to silently
        # produce wrong scores.
        raise SystemExit(
            "Checkpoint has ema_zscore_normalize=True; this script's "
            "all_scores recompute would not match what the router used."
        )
    model = build_model()
    ckpt = run_dir / f"step{step}" / "model_and_optim"
    print(f"Loading checkpoint: {ckpt}")
    load_model_and_optim_state(str(ckpt), model, strict=False)
    model.eval()
    model.to(device=device, dtype=dtype)
    return model


# ── titles source ─────────────────────────────────────────────────────────────


def iter_titles_from_file(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                yield t


def iter_titles_from_hf(hf_id: str, hf_config: str) -> Iterator[str]:
    """Stream titles from HuggingFace `wikimedia/wikipedia` (no full download)."""
    from datasets import load_dataset  # type: ignore

    ds = load_dataset(hf_id, hf_config, split="train", streaming=True)
    for row in ds:
        t = row.get("title")
        if isinstance(t, str) and t:
            yield t


# ── batching ──────────────────────────────────────────────────────────────────


def tokenize_and_batch(
    titles_iter: Iterator[str],
    tokenizer,
    max_titles: int,
    batch_size: int,
    max_len: int,
    pad_id: int,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
    """Yield (input_ids[B,S], lengths[B], titles[B]) batches, right-padded to per-batch max."""
    buf_ids: List[List[int]] = []
    buf_titles: List[str] = []
    n_emitted = 0
    for title in titles_iter:
        if n_emitted >= max_titles:
            break
        ids = tokenizer.encode(title, add_special_tokens=False)
        if not ids:
            continue
        if len(ids) > max_len:
            ids = ids[:max_len]
        buf_ids.append(ids)
        buf_titles.append(title)
        n_emitted += 1
        if len(buf_ids) == batch_size:
            yield _pad_batch(buf_ids, pad_id) + (buf_titles,)
            buf_ids, buf_titles = [], []
    if buf_ids:
        yield _pad_batch(buf_ids, pad_id) + (buf_titles,)


def _pad_batch(rows: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rows_t = [torch.tensor(r, dtype=torch.long) for r in rows]
    input_ids = pad_sequence(rows_t, batch_first=True, padding_value=pad_id)
    lengths = torch.tensor([len(r) for r in rows], dtype=torch.long)
    return input_ids, lengths


# ── hooks ─────────────────────────────────────────────────────────────────────


class RouterCapture:
    """Per-batch buffer: layer_idx -> (h_in, expert_indices, expert_weights)."""

    def __init__(self):
        self.per_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    def reset(self):
        self.per_layer.clear()


def make_router_hook(layer_idx: int, capture: RouterCapture):
    def hook(module, args, output):
        x = args[0].detach()
        # MoERouter.forward returns (expert_weights, expert_indices, batch_size_per_expert, aux_loss)
        capture.per_layer[layer_idx] = (x, output[1].detach(), output[0].detach())

    return hook


def attach_hooks(model: torch.nn.Module, capture: RouterCapture) -> List[torch.utils.hooks.RemovableHandle]:
    handles = []
    moe_layer_idx = 0
    for block in model.blocks.values():
        moe = getattr(block, "feed_forward_moe", None)
        if moe is None:
            continue
        handles.append(moe.router.register_forward_hook(make_router_hook(moe_layer_idx, capture)))
        moe_layer_idx += 1
    if moe_layer_idx == 0:
        raise RuntimeError("no MoE layers found")
    return handles


# ── per-batch extraction ──────────────────────────────────────────────────────


def collect_moe_layers(model: torch.nn.Module):
    """List of (router_module, w_gate[E,H,D]) in layer order."""
    out = []
    for block in model.blocks.values():
        moe = getattr(block, "feed_forward_moe", None)
        if moe is None:
            continue
        E = moe.num_experts
        H = moe.experts.mlp.hidden_size
        D = moe.experts.mlp.d_model
        w_gate = moe.experts.mlp.w1.detach().view(E, H, D)
        out.append((moe.router, w_gate))
    return out


def extract_for_last_token(
    capture: RouterCapture,
    lengths: torch.Tensor,
    moe_layers,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns per batch (B):
      expert_indices       : (B, L, K)        uint8
      expert_scores        : (B, L, K)        fp16 — model's own post-renorm gather weights
      all_scores           : (B, L, E)        fp16 — full sigmoid(h @ Wᵀ) over all experts
      w_gate_top10_idx     : (B, L, K, 10)    int16
      w_gate_top10_val     : (B, L, K, 10)    fp32 — the inner products at those indices
    """
    L = len(moe_layers)
    B = lengths.shape[0]
    device = capture.per_layer[0][0].device
    last_idx = (lengths - 1).to(device)

    E = moe_layers[0][1].shape[0]

    expert_indices = np.empty((B, L, top_k), dtype=np.uint8)
    expert_scores = np.empty((B, L, top_k), dtype=np.float16)
    all_scores = np.empty((B, L, E), dtype=np.float16)
    w_gate_top10_idx = np.empty((B, L, top_k, TOP10), dtype=np.int16)
    w_gate_top10_val = np.empty((B, L, top_k, TOP10), dtype=np.float32)

    batch_arange = torch.arange(B, device=device)

    for L_idx, (router, w_gate) in enumerate(moe_layers):
        h_in, ei, ew = capture.per_layer[L_idx]
        h_last = h_in[batch_arange, last_idx, :].float()

        # Use the router's own logit computation so any future change to the
        # linear router's path (e.g. learned bias) is picked up automatically.
        # Sigmoid matches `MoERouter.forward` for the deepseek variant. The
        # +1e-7 the router adds (router.py:993) is irrelevant for argsort and
        # is below fp16 precision anyway.
        logits = router.get_expert_logits(h_last).float()
        scores = torch.sigmoid(logits)

        ei_last = ei[batch_arange, last_idx, :]
        ew_last = ew[batch_arange, last_idx, :].float()

        expert_indices[:, L_idx, :] = ei_last.to(torch.uint8).cpu().numpy()
        expert_scores[:, L_idx, :] = ew_last.cpu().numpy().astype(np.float16)
        all_scores[:, L_idx, :] = scores.cpu().numpy().astype(np.float16)

        # Gather chosen-expert gates first, then cast — keeps the (E,H,D)
        # full-layer tensor in bf16 and only materialises (B,K,H,D) in fp32.
        chosen_w = w_gate[ei_last.long()].float()             # (B, K, H, D)
        proj = torch.einsum("bkhd,bd->bkh", chosen_w, h_last)  # (B, K, H)
        top = proj.topk(TOP10, dim=-1)
        w_gate_top10_idx[:, L_idx, :, :] = top.indices.to(torch.int16).cpu().numpy()
        w_gate_top10_val[:, L_idx, :, :] = top.values.cpu().numpy()

    return expert_indices, expert_scores, all_scores, w_gate_top10_idx, w_gate_top10_val


# ── main loop ─────────────────────────────────────────────────────────────────


def _save_npz(path: Path, ei, es, asn, w10i, w10v, titles, **extra):
    np.savez_compressed(
        path,
        expert_indices=np.concatenate(ei, axis=0),
        expert_scores=np.concatenate(es, axis=0),
        all_scores=np.concatenate(asn, axis=0),
        w_gate_top10_idx=np.concatenate(w10i, axis=0),
        w_gate_top10_val=np.concatenate(w10v, axis=0),
        titles=np.array(titles, dtype=object),
        **extra,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="moe-1b-269440-deepseek", help="run dir under runs/")
    p.add_argument("--step", type=int, required=True, help="checkpoint step (e.g. 39750)")
    p.add_argument("--out", required=True, help=".npz output path")
    p.add_argument("--max-titles", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-len", type=int, default=64, help="truncate titles to this many tokens")
    p.add_argument("--titles-file", default=None, help="optional: one title per line; otherwise stream from HF")
    p.add_argument("--hf-dataset", default="wikimedia/wikipedia")
    p.add_argument("--hf-config", default="20231101.en")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--save-every", type=int, default=64, help="flush partial results every N batches")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"device={device}  dtype={dtype}")

    run_dir = RUNS_DIR / args.run
    if not run_dir.is_dir():
        raise SystemExit(f"run not found: {run_dir}")

    from transformers import AutoTokenizer  # type: ignore

    tcfg = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    tok = AutoTokenizer.from_pretrained(tcfg.identifier)
    pad_id = tcfg.pad_token_id

    model = load_model(run_dir, args.step, device, dtype)
    capture = RouterCapture()
    handles = attach_hooks(model, capture)
    moe_layers = collect_moe_layers(model)
    top_k = moe_layers[0][0].top_k
    E, H, D = moe_layers[0][1].shape
    print(f"MoE layers: {len(moe_layers)}  experts: {E}  top_k: {top_k}  H: {H}  D: {D}")

    titles_iter = (
        iter_titles_from_file(Path(args.titles_file))
        if args.titles_file
        else iter_titles_from_hf(args.hf_dataset, args.hf_config)
    )
    print(f"titles source: {args.titles_file or f'HF {args.hf_dataset}/{args.hf_config}'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    chunks_ei: List[np.ndarray] = []
    chunks_es: List[np.ndarray] = []
    chunks_as: List[np.ndarray] = []
    chunks_w10i: List[np.ndarray] = []
    chunks_w10v: List[np.ndarray] = []
    accepted_titles: List[str] = []

    partial_path = Path(str(args.out) + ".partial.npz")
    n_done = 0
    t0 = time.time()
    try:
        with torch.inference_mode():
            for b_idx, (input_ids, lengths, titles_b) in enumerate(
                tokenize_and_batch(titles_iter, tok, args.max_titles, args.batch_size, args.max_len, pad_id)
            ):
                input_ids = input_ids.to(device)
                capture.reset()
                _ = model(input_ids)
                ei, es, asn, w10i, w10v = extract_for_last_token(capture, lengths.to(device), moe_layers, top_k)
                chunks_ei.append(ei)
                chunks_es.append(es)
                chunks_as.append(asn)
                chunks_w10i.append(w10i)
                chunks_w10v.append(w10v)
                accepted_titles.extend(titles_b)
                n_done += ei.shape[0]
                if b_idx % 10 == 0:
                    rate = n_done / max(time.time() - t0, 1e-6)
                    print(f"  batch {b_idx:5d}  titles={n_done:>7d}  ({rate:6.1f}/s)")
                if (b_idx + 1) % args.save_every == 0:
                    _save_npz(partial_path, chunks_ei, chunks_es, chunks_as,
                              chunks_w10i, chunks_w10v, accepted_titles, n_done=n_done)
    finally:
        for h in handles:
            h.remove()

    print(f"Done {n_done} titles in {time.time() - t0:.1f}s; saving to {args.out}")
    _save_npz(Path(args.out), chunks_ei, chunks_es, chunks_as,
              chunks_w10i, chunks_w10v, accepted_titles,
              run=args.run, step=args.step)
    if partial_path.exists():
        partial_path.unlink()
    print("OK")


if __name__ == "__main__":
    main()
