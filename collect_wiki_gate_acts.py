#!/usr/bin/env python3
"""
Collect per-neuron gate pre-activations and routing metadata from a deepseek MoE
checkpoint over the Wikipedia tokenized dataset, for the geometric-coupling
analysis in the NeurIPS paper.

Per layer l in 0..N_LAYERS-1, the output .npz contains:
  l{l}_gate_preact   float16 (N, top_k, hidden)  h @ w1[k].T per (token, rank)
  l{l}_expert_idx    int8    (N, top_k)          k = expert at column r (rank r+1)
  l{l}_expert_score  float16 (N, top_k)          L1-normalised UNBIASED sigmoid
                                                 score at rank r — column order is
                                                 by `score+bias` (biased topk),
                                                 so column 0 is NOT necessarily
                                                 the largest unbiased score.
  l{l}_router_weight float32 (n_experts, d_model) router linear weights, for
                                                 recomputing raw scores post-hoc
  l{l}_router_bias   float32 (n_experts,)         deepseek bias rule values
                                                 (omitted if router has no bias)
  l{l}_expert_mean   float32 (n_experts, hidden)  per-expert per-neuron mean of
                                                 gate_preact across all (token,
                                                 rank) pairs that selected k
  l{l}_expert_std    float32 (n_experts, hidden)  per-expert per-neuron std
  l{l}_expert_count  int64   (n_experts,)         total (token,rank) selections
                                                 per expert. Sum = N * top_k
                                                 (each token contributes top_k).

Globals: n_tokens, n_seqs, step.

Why we recompute h @ w1[k].T offline (rather than hook the model's gmm output):
the dispatch path permutes tokens by expert before calling gmm, so reconstructing
the (token, rank) layout from the gmm output is more error-prone than just doing
the matmul against the captured router input. The numbers match the model's
internal gate value to bfloat16 precision (gmm casts inputs to bf16); we use
mlp.gmm() so the math goes through the same kernel the model uses.

Usage:
    python collect_wiki_gate_acts.py [--max-seqs 64] [--batch-size 4]
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Match training-time grouped_mm setup (see moe-1b.slurm); without this the
# DroplessMoEMLP falls back to a slow Python loop in gmm().
os.environ.setdefault("OLMO_CORE_USE_TORCH_GROUPED_MM", "1")

import numpy as np
import torch

REPO_ROOT = Path(__file__).parent.resolve()
CKPT_PATH = REPO_ROOT / "runs/moe-1b-269440-deepseek/step21000/model_and_optim"
STEP = 21000

N_LAYERS = 9
N_EXPERTS = 64
TOP_K = 6
D_MODEL = 1024
HIDDEN_SIZE = 512
SEQ_LEN = 2048


def _load_moe1b():
    # The training script's filename has a hyphen so it can't be plain-imported;
    # importlib + sys.modules lets Config deserialisation resolve dataclasses
    # defined inside it (otherwise `merge([])` raises ModuleNotFoundError).
    spec = importlib.util.spec_from_file_location(
        "moe_1b", REPO_ROOT / "src/scripts/train/moe-1b.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_1b"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(device: torch.device):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    m = _load_moe1b()
    config = m.build_config("collect", m.RoutingVariant.deepseek, [])
    print("Building model...")
    model = config.model.build(init_device="cpu")
    print(f"Loading checkpoint from {CKPT_PATH} ...")
    load_model_and_optim_state(str(CKPT_PATH), model, work_dir="/tmp/ckpt_work")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, m.DATASET_DIR


def assert_checkpoint_loaded(model):
    """Catch silent checkpoint-load failures: post-training w1.std is well above init."""
    for i in range(N_LAYERS):
        w1 = model.blocks[str(i)].feed_forward_moe.experts.mlp.w1
        s = w1.float().std().item()
        if s < 0.01:
            raise RuntimeError(
                f"layer {i} w1.std={s:.4f} — too close to init_std=0.02. "
                "Checkpoint likely failed to load."
            )
    print(f"  w1.std per layer: {[round(model.blocks[str(i)].feed_forward_moe.experts.mlp.w1.float().std().item(), 4) for i in range(N_LAYERS)]}")


def register_hooks(model):
    """Capture router input (h) and output (weights, indices) per layer, on-device."""
    h_cache: dict[int, torch.Tensor] = {}
    idx_cache: dict[int, torch.Tensor] = {}
    score_cache: dict[int, torch.Tensor] = {}
    handles = []

    for i in range(N_LAYERS):
        router = model.blocks[str(i)].feed_forward_moe.router

        def _pre(_mod, inp, *, _i=i):
            h_cache[_i] = inp[0].reshape(-1, D_MODEL).detach()

        def _post(_mod, _inp, out, *, _i=i):
            weights, indices, _, _ = out
            score_cache[_i] = weights.reshape(-1, TOP_K).detach()
            idx_cache[_i] = indices.reshape(-1, TOP_K).detach().long()

        handles.append(router.register_forward_pre_hook(_pre))
        handles.append(router.register_forward_hook(_post))

    return h_cache, idx_cache, score_cache, handles


def gate_preacts_via_gmm(h, expert_indices, mlp):
    """
    Compute gate pre-activations for each (token, rank) using the model's own
    grouped_mm, so the precision (bfloat16) and reduction order match what the
    model actually runs at inference.

    h:              (N, d_model)
    expert_indices: (N, top_k) long
    mlp:            DroplessMoEMLP for this layer
    Returns:        (N, top_k, hidden_size) in bf16
    """
    N = h.shape[0]
    h_expanded = h.unsqueeze(1).expand(N, TOP_K, D_MODEL).reshape(-1, D_MODEL)  # (N*K, D)
    flat_idx = expert_indices.reshape(-1)                                       # (N*K,)

    sorted_idx, perm = flat_idx.sort(stable=True)
    sorted_h = h_expanded[perm]
    batch_sizes = torch.bincount(sorted_idx, minlength=N_EXPERTS)

    w1 = mlp.w1.view(N_EXPERTS, HIDDEN_SIZE, D_MODEL)
    sorted_gate = mlp.gmm(sorted_h, w1, batch_sizes, trans_b=True)  # (N*K, H)

    inv_perm = perm.argsort()
    return sorted_gate[inv_perm].view(N, TOP_K, HIDDEN_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seqs",   type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", default="wiki_gate_acts_deepseek_step21k.npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, dataset_dir = build_model(device)
    assert_checkpoint_loaded(model)

    h_cache, idx_cache, score_cache, handles = register_hooks(model)

    # Wikipedia files are raw uint16 (not standard .npy headers); see
    # olmo_core.data.utils.load_array_slice for the same convention.
    wiki_paths = [
        f"{dataset_dir}/wiki/part-0-00000.npy",
        f"{dataset_dir}/wiki/part-0-00001.npy",
    ]
    parts = [np.memmap(p, dtype=np.uint16, mode="r") for p in wiki_paths]
    tokens_flat = np.concatenate(parts)
    n_seqs = min(args.max_seqs, len(tokens_flat) // SEQ_LEN)
    seqs = tokens_flat[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)
    print(f"Will process {n_seqs} sequences x {SEQ_LEN} tokens = {n_seqs * SEQ_LEN:,} tokens")

    # Per-layer accumulators. Stats live on GPU (index_add_ is much faster than
    # the per-expert numpy mask loop) and are moved to CPU once at the end.
    gate_chunks: dict[int, list[np.ndarray]]  = {i: [] for i in range(N_LAYERS)}
    idx_chunks:  dict[int, list[np.ndarray]]  = {i: [] for i in range(N_LAYERS)}
    score_chunks:dict[int, list[np.ndarray]]  = {i: [] for i in range(N_LAYERS)}
    exp_sum    = {i: torch.zeros(N_EXPERTS, HIDDEN_SIZE, device=device, dtype=torch.float64) for i in range(N_LAYERS)}
    exp_sum_sq = {i: torch.zeros(N_EXPERTS, HIDDEN_SIZE, device=device, dtype=torch.float64) for i in range(N_LAYERS)}
    exp_count  = {i: torch.zeros(N_EXPERTS, device=device, dtype=torch.int64) for i in range(N_LAYERS)}

    bs = args.batch_size
    n_batches = (n_seqs + bs - 1) // bs
    for bi in range(n_batches):
        start = bi * bs
        end = min(start + bs, n_seqs)
        batch = torch.from_numpy(seqs[start:end].astype(np.int64)).to(device)

        with torch.no_grad():
            model(batch)

            for i in range(N_LAYERS):
                h = h_cache[i].float()
                expert_indices = idx_cache[i]
                mlp = model.blocks[str(i)].feed_forward_moe.experts.mlp

                gate = gate_preacts_via_gmm(h, expert_indices, mlp).float()  # (N, K, H)

                # Persist per-batch chunks (CPU, fp16 for storage)
                gate_chunks[i].append(gate.to(torch.float16).cpu().numpy())
                idx_chunks[i].append(expert_indices.to(torch.int8).cpu().numpy())
                score_chunks[i].append(score_cache[i].to(torch.float16).cpu().numpy())

                # Vectorised per-expert running stats on GPU.
                k_flat = expert_indices.reshape(-1)
                g_flat = gate.reshape(-1, HIDDEN_SIZE).double()
                exp_sum[i].index_add_(0, k_flat, g_flat)
                exp_sum_sq[i].index_add_(0, k_flat, g_flat * g_flat)
                exp_count[i] += torch.bincount(k_flat, minlength=N_EXPERTS)

        if bi % 4 == 0 or bi == n_batches - 1:
            print(f"  [{end}/{n_seqs} seqs]")

    for hh in handles:
        hh.remove()

    # Sum-of-counts assertion: every (token, rank) pair must contribute exactly once
    expected_count = n_seqs * SEQ_LEN * TOP_K
    for i in range(N_LAYERS):
        actual = exp_count[i].sum().item()
        assert actual == expected_count, (
            f"layer {i}: expert_count sum {actual} != expected {expected_count} "
            "(some hooks dropped data)"
        )

    print(f"\nSaving to {args.out} ...")
    out: dict[str, np.ndarray] = {
        "n_tokens": np.array(n_seqs * SEQ_LEN, dtype=np.int64),
        "n_seqs":   np.array(n_seqs, dtype=np.int64),
        "step":     np.array(STEP, dtype=np.int64),
    }

    for i in range(N_LAYERS):
        cnt    = exp_count[i].cpu().numpy()
        safe   = np.maximum(cnt, 1)[:, None]
        sum_i  = exp_sum[i].cpu().numpy()
        sumsq_i = exp_sum_sq[i].cpu().numpy()
        mean_i = (sum_i / safe).astype(np.float32)
        var_i  = sumsq_i / safe - mean_i.astype(np.float64) ** 2
        std_i  = np.sqrt(np.maximum(var_i, 1e-8)).astype(np.float32)

        router = model.blocks[str(i)].feed_forward_moe.router
        out[f"l{i}_router_weight"] = (
            router.weight.detach().reshape(N_EXPERTS, D_MODEL).float().cpu().numpy()
        )
        if getattr(router, "score_bias", None) is not None:
            out[f"l{i}_router_bias"] = router.score_bias.detach().float().cpu().numpy()

        out[f"l{i}_gate_preact"]  = np.concatenate(gate_chunks[i],  axis=0)
        out[f"l{i}_expert_idx"]   = np.concatenate(idx_chunks[i],   axis=0)
        out[f"l{i}_expert_score"] = np.concatenate(score_chunks[i], axis=0)
        out[f"l{i}_expert_mean"]  = mean_i
        out[f"l{i}_expert_std"]   = std_i
        out[f"l{i}_expert_count"] = cnt

        print(f"  Layer {i}: {out[f'l{i}_gate_preact'].shape}, count [{cnt.min()}, {cnt.max()}]")

    # np.savez (uncompressed) — fp16 gate vectors are high-entropy so deflate
    # only saves ~0% but takes much longer.
    np.savez(args.out, **out)
    print(f"Saved {Path(args.out).stat().st_size / 1e9:.2f} GB to {args.out}")


if __name__ == "__main__":
    main()
