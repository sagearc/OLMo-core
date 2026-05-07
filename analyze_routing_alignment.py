#!/usr/bin/env python3
"""
Routing alignment: does each routing rank position correspond to the highest dot-product
similarity expert?

For a fixed eval batch, at each MoE layer, for each token:
  - Get hidden state h entering the router
  - Get the actual top-6 routing decision (expert indices, sorted by routing score descending)
  - Compute dot product  dp_k = h · gate_k  for all 64 experts
    (gate_k = weight row for linear router, centroid vector for centroid router)
  - Record dp at routing rank positions 1–6

If routing follows dot-product similarity, the curve is monotonically decreasing.

Usage:
    python analyze_routing_alignment.py \
        --runs moe-1b-283608-ema_centroid moe-1b-284218-baseline_no_loss \
        [--step STEP] [--n-batches N] [--layer LAYER] [--out path.png]
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from olmo_core.data import NumpyDataLoaderConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyFSLDatasetConfig, NumpyPaddedFSLDatasetConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.moe.router import MoECentroidRouter, MoELinearRouter
from olmo_core.nn.transformer import TransformerConfig

RUNS_DIR = Path(__file__).parent / "runs"
EVAL_BASE_DIR = os.environ.get("OLMO_EVAL_BASE_DIR", "dataset/olmoe-1pct")
REPO_DIR = str(Path(__file__).parent)
SEQ_LEN = 2048
TOP_K = 6
NUM_EXPERTS = 64
D_MODEL = 1024


# ── model building ─────────────────────────────────────────────────────────────

def detect_router_type(run_dir: Path) -> str:
    """Read config.json and return 'centroid' or 'default'."""
    import json
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        # try latest step
        for step_dir in sorted(run_dir.iterdir()):
            cfg_path = step_dir / "config.json"
            if cfg_path.exists():
                break
    with open(cfg_path) as f:
        cfg = json.load(f)
    router_name = (
        cfg["model"]["block"]["feed_forward_moe"]["router"]["name"]
    )
    return router_name  # "centroid" or "default"


def build_model(router_type: str) -> torch.nn.Module:
    """Build a plain (non-FSDP, non-compiled) transformer matching the training config."""
    from olmo_core.nn.moe import MoEConfig, MoERouterGatingFunction, MoERouterType
    from olmo_core.nn.moe.router import MoERouterConfig

    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()

    model_config = TransformerConfig.llama_like_moe(
        d_model=D_MODEL,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=9,
        n_heads=8,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        expert_hidden_size=int(0.5 * D_MODEL),
        shared_expert_hidden_size=2 * int(0.5 * D_MODEL),
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        lb_loss_weight=None,
        z_loss_weight=None,
        init_std=0.02,
    )

    if router_type == "centroid":
        from olmo_core.nn.transformer.config import TransformerBlockConfig
        from typing import cast
        block = cast(TransformerBlockConfig, model_config.block)
        moe = block.feed_forward_moe
        assert moe is not None
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_lr_lambda = 1.0
        moe.router.gating_function = MoERouterGatingFunction.identity

    model = model_config.build(init_device="cpu")
    return model


def latest_step(run_dir: Path) -> int:
    steps = [
        int(n[4:])
        for n in os.listdir(run_dir)
        if n.startswith("step") and (run_dir / n).is_dir() and n[4:].isdigit()
    ]
    return max(steps)


def load_model(run_dir: Path, step: int) -> torch.nn.Module:
    router_type = detect_router_type(run_dir)
    print(f"  router type: {router_type}")
    model = build_model(router_type)
    ckpt_dir = run_dir / f"step{step}" / "model_and_optim"
    print(f"  loading checkpoint from {ckpt_dir} ...")
    load_model_and_optim_state(str(ckpt_dir), model, strict=False)
    model.eval()
    return model


# ── eval data ──────────────────────────────────────────────────────────────────

def get_eval_batches(n_batches: int, batch_size: int = 4) -> List[torch.Tensor]:
    """Load n_batches × batch_size sequences from the validation mix."""
    from olmo_core.data.mixes import DataMix
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    dataset = NumpyPaddedFSLDatasetConfig.from_data_mix(
        DataMix.v3_small_ppl_validation,
        mix_base_dir=EVAL_BASE_DIR,
        sequence_length=SEQ_LEN,
        tokenizer=tokenizer,
        work_dir=f"{REPO_DIR}/dataset-cache",
    ).build()

    batches = []
    indices = list(range(n_batches * batch_size))
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        seqs = [torch.tensor(dataset[j]["input_ids"], dtype=torch.long) for j in batch_indices]
        batches.append(torch.stack(seqs))  # (B, S)
    return batches


# ── alignment measurement ──────────────────────────────────────────────────────

def measure_alignment(
    model: torch.nn.Module,
    batches: List[torch.Tensor],
    layer: Optional[int],
    device: torch.device,
) -> np.ndarray:
    """
    Returns array of shape (TOP_K,): mean dot product at each routing rank position.
    Averaged over all tokens, all MoE layers (or a single layer if specified).
    """
    model = model.to(device)

    # Accumulators: sum of dp scores at each rank, and count
    dp_sum = np.zeros(TOP_K, dtype=np.float64)
    dp_count = np.zeros(TOP_K, dtype=np.int64)

    hooks = []
    # We'll collect (h, expert_indices) per layer per forward call
    captures: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

    def make_hook(layer_idx: int, router):
        def hook(module, args, output):
            # args[0] is x entering the router: shape (B, S, D) or (B*S, D)
            x = args[0].detach().float()
            # output: (expert_weights, expert_indices, batch_size_per_expert, aux_loss)
            expert_indices = output[1].detach()  # (B, S, TOP_K) or (N, TOP_K)

            # flatten to (N, D) and (N, TOP_K)
            h = x.view(-1, D_MODEL)
            idx = expert_indices.view(-1, TOP_K)

            # get gate rows: centroids or weight rows
            if isinstance(module, MoECentroidRouter):
                gate_rows = F.normalize(module._centroid.detach().float(), dim=-1)  # (E, D)
            else:
                gate_rows = module.weight.detach().float().view(NUM_EXPERTS, D_MODEL)  # (E, D)

            # dot product of each token with all gate rows: (N, E)
            dp = h @ gate_rows.t()

            # for each token, gather dp at each routing rank position
            # idx[:, 0] = rank-1 expert, idx[:, 1] = rank-2, etc.
            for rank in range(TOP_K):
                expert_at_rank = idx[:, rank].long()  # (N,)
                scores_at_rank = dp[torch.arange(h.shape[0]), expert_at_rank]  # (N,)
                dp_sum[rank] += scores_at_rank.sum().item()
                dp_count[rank] += h.shape[0]

        return hook

    # Register hooks on all (or specified) MoE router modules
    for block_idx, block in enumerate(model.blocks.values()):
        if layer is not None and block_idx != layer:
            continue
        moe = getattr(block, "feed_forward_moe", None)
        if moe is None:
            continue
        router = moe.router
        h = router.register_forward_hook(make_hook(block_idx, router))
        hooks.append(h)

    try:
        with torch.no_grad():
            for batch in batches:
                batch = batch.to(device)
                model(batch)
    finally:
        for h in hooks:
            h.remove()

    # mean dp at each rank
    mean_dp = dp_sum / np.maximum(dp_count, 1)
    return mean_dp


# ── plot ───────────────────────────────────────────────────────────────────────

def make_plot(results: Dict[str, np.ndarray], out_path: str, layer: Optional[int]):
    ranks = list(range(1, TOP_K + 1))
    fig, ax = plt.subplots(figsize=(7, 5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (label, mean_dp) in enumerate(results.items()):
        # normalize so rank-1 = 1.0 for fair visual comparison
        norm = mean_dp / (mean_dp[0] + 1e-12)
        ax.plot(ranks, norm, "o-", color=colors[i % len(colors)], label=label, linewidth=2, markersize=6)

    ax.set_xlabel("Routing rank position", fontsize=12)
    ax.set_ylabel("Mean dot-product score (normalized to rank 1 = 1.0)", fontsize=11)
    layer_str = f"layer {layer}" if layer is not None else "all layers"
    ax.set_title(f"Routing alignment: dot-product score vs. routing rank\n({layer_str})", fontsize=12)
    ax.set_xticks(ranks)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--runs", nargs="+", required=True,
        help="Run names under runs/, e.g. moe-1b-283608-ema_centroid moe-1b-284218-baseline_no_loss"
    )
    p.add_argument("--step", type=int, default=None, help="Checkpoint step (default: latest)")
    p.add_argument("--n-batches", type=int, default=8, help="Number of eval batches (default: 8)")
    p.add_argument("--batch-size", type=int, default=4, help="Sequences per batch (default: 4)")
    p.add_argument("--layer", type=int, default=None, help="Single MoE layer to analyze (default: all)")
    p.add_argument("--out", default="routing_alignment.png")
    p.add_argument("--cpu", action="store_true", help="Force CPU (default: use CUDA if available)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    print("Loading eval batches ...")
    batches = get_eval_batches(args.n_batches, args.batch_size)
    print(f"  {len(batches)} batches × {batches[0].shape[0]} seqs × {SEQ_LEN} tokens")

    results = {}
    for run_name in args.runs:
        run_dir = RUNS_DIR / run_name
        if not run_dir.is_dir():
            sys.exit(f"Run not found: {run_dir}")
        step = args.step if args.step is not None else latest_step(run_dir)
        print(f"\n── {run_name}  step={step} ──")
        model = load_model(run_dir, step)
        mean_dp = measure_alignment(model, batches, args.layer, device)
        print(f"  mean dp at ranks 1-6: {np.round(mean_dp, 4)}")
        # label: strip the job-id prefix for readability
        label = run_name.split("-")[-1] + f" (step {step})"
        results[label] = mean_dp
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    make_plot(results, args.out, args.layer)


if __name__ == "__main__":
    main()
