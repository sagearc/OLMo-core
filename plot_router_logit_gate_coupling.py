#!/usr/bin/env python3
"""Plot router score vs selected-expert response without decile/bin markers."""
from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OLMO_CORE_USE_TORCH_GROUPED_MM", "1")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

REPO_ROOT = Path(__file__).parent.resolve()
RUNS_ROOT = REPO_ROOT / "runs"

N_LAYERS = 9
N_EXPERTS = 64
TOP_K = 6
D_MODEL = 1024
HIDDEN_SIZE = 512
SEQ_LEN = 2048

TOL_BLUE = "#4477AA"

sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_moe1b():
    spec = importlib.util.spec_from_file_location(
        "moe_1b", REPO_ROOT / "src/scripts/train/moe-1b.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_1b"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_model(run: str, step: int, device: torch.device):
    from olmo_core.distributed.checkpoint import load_model_and_optim_state

    train = load_moe1b()
    config = train.build_config("native-coupling", train.RoutingVariant.deepseek, [])
    model = config.model.build(init_device="cpu")
    ckpt = RUNS_ROOT / run / f"step{step}" / "model_and_optim"
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    print(f"Loading checkpoint: {ckpt}")
    load_model_and_optim_state(str(ckpt), model, work_dir="/tmp/ckpt_work_native_coupling")
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, train.DATASET_DIR


def register_router_hooks(model):
    h_cache: dict[int, torch.Tensor] = {}
    idx_cache: dict[int, torch.Tensor] = {}
    handles = []

    for layer in range(N_LAYERS):
        router = model.blocks[str(layer)].feed_forward_moe.router

        def pre_hook(_mod, inp, *, layer_idx=layer):
            h_cache[layer_idx] = inp[0].reshape(-1, D_MODEL).detach()

        def post_hook(_mod, _inp, out, *, layer_idx=layer):
            _weights, indices, _batch_size_per_expert, _aux_loss = out
            idx_cache[layer_idx] = indices.reshape(-1, TOP_K).detach().long()

        handles.append(router.register_forward_pre_hook(pre_hook))
        handles.append(router.register_forward_hook(post_hook))

    return h_cache, idx_cache, handles


def gate_preacts_via_gmm(h: torch.Tensor, expert_indices: torch.Tensor, mlp) -> torch.Tensor:
    n_tokens = h.shape[0]
    h_expanded = h.unsqueeze(1).expand(n_tokens, TOP_K, D_MODEL).reshape(-1, D_MODEL)
    flat_idx = expert_indices.reshape(-1)

    sorted_idx, perm = flat_idx.sort(stable=True)
    sorted_h = h_expanded[perm]
    batch_sizes = torch.bincount(sorted_idx, minlength=N_EXPERTS)

    w1 = mlp.w1.view(N_EXPERTS, HIDDEN_SIZE, D_MODEL)
    sorted_gate = mlp.gmm(sorted_h, w1, batch_sizes, trans_b=True)
    inv_perm = perm.argsort()
    return sorted_gate[inv_perm].view(n_tokens, TOP_K, HIDDEN_SIZE)


def zscore_by_expert(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat_idx = idx.reshape(-1).astype(np.intp)
    flat_val = values.reshape(-1).astype(np.float64)
    counts = np.bincount(flat_idx, minlength=N_EXPERTS)
    safe_counts = np.maximum(counts, 1)
    sums = np.bincount(flat_idx, weights=flat_val, minlength=N_EXPERTS)
    sums_sq = np.bincount(flat_idx, weights=flat_val * flat_val, minlength=N_EXPERTS)
    mean = sums / safe_counts
    var = np.maximum(sums_sq / safe_counts - mean * mean, 1e-8)
    std = np.sqrt(var)
    return ((flat_val - mean[flat_idx]) / std[flat_idx]).reshape(values.shape).astype(np.float32)


def rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks_sorted = np.empty(len(x), dtype=np.float64)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
        start = end
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = ranks_sorted
    return ranks


def spearmanr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask].astype(np.float64, copy=False)
    y = y[mask].astype(np.float64, copy=False)
    if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return np.nan
    rx = rankdata_average(x)
    ry = rankdata_average(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    return float(np.dot(rx, ry) / denom) if denom > 0 else np.nan


def layer_expert_rhos(raw_logits: np.ndarray, response: np.ndarray, idx: np.ndarray) -> np.ndarray:
    flat_logit = raw_logits.reshape(-1).astype(np.float64)
    flat_response = response.reshape(-1).astype(np.float64)
    flat_idx = idx.reshape(-1).astype(np.intp)
    rhos = []
    for expert in range(N_EXPERTS):
        sel = flat_idx == expert
        if sel.sum() >= 128:
            rho = spearmanr(flat_logit[sel], flat_response[sel])
            if np.isfinite(rho):
                rhos.append(rho)
    return np.array(rhos, dtype=np.float64)


def binomial_two_sided_p_value(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial p-value, computed in log space."""
    if n == 0:
        return np.nan
    if not 0.0 < p0 < 1.0:
        raise ValueError(f"p0 must be in (0, 1), got {p0}")

    def log_pmf(i: int) -> float:
        return (
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            + i * math.log(p0)
            + (n - i) * math.log1p(-p0)
        )

    if k >= n * p0:
        lo, hi = k, n
    else:
        lo, hi = 0, k
    logs = np.array([log_pmf(i) for i in range(lo, hi + 1)], dtype=np.float64)
    m = logs.max()
    one_sided = float(np.exp(m) * np.exp(logs - m).sum())
    return min(1.0, 2.0 * one_sided)


def format_p_value(p: float) -> str:
    if not np.isfinite(p):
        return "n/a"
    if p == 0.0:
        return "<1e-300"
    if p < 1e-3:
        exponent = int(np.floor(np.log10(p)))
        mantissa = p / (10.0 ** exponent)
        return f"{mantissa:.1f}e{exponent}"
    return f"{p:.3f}"


def binned_stats(x: np.ndarray, y: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    flat_x = x.reshape(-1).astype(np.float64)
    flat_y = y.reshape(-1).astype(np.float64)
    mask = np.isfinite(flat_x) & np.isfinite(flat_y)
    flat_x = flat_x[mask]
    flat_y = flat_y[mask]

    edges = np.quantile(flat_x, np.linspace(0, 1, n_bins + 1))
    bins = np.searchsorted(edges[1:-1], flat_x, side="right")

    x_mean = np.empty(n_bins, dtype=np.float64)
    y_mean = np.empty(n_bins, dtype=np.float64)
    y_median = np.empty(n_bins, dtype=np.float64)
    y_p25 = np.empty(n_bins, dtype=np.float64)
    y_p75 = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=np.int64)

    for b in range(n_bins):
        sel = bins == b
        counts[b] = int(sel.sum())
        if counts[b] == 0:
            raise ValueError(f"empty router-score group {b}; check x distribution")
        x_mean[b] = flat_x[sel].mean()
        y_mean[b] = flat_y[sel].mean()
        y_median[b] = np.median(flat_y[sel])
        y_p25[b], y_p75[b] = np.percentile(flat_y[sel], [25, 75])

    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "y_median": y_median,
        "y_p25": y_p25,
        "y_p75": y_p75,
        "counts": counts,
    }


def low_high_summary(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    flat_x = x.reshape(-1)
    flat_y = y.reshape(-1)
    return float(np.median(flat_y[flat_x < 0])), float(np.median(flat_y[flat_x > 0]))


def plot(stats: dict[str, np.ndarray], path: Path, *, median_rho: float, p_value: float):
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    ax.fill_between(
        stats["x_mean"],
        stats["y_p25"],
        stats["y_p75"],
        color=TOL_BLUE,
        alpha=0.08,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        stats["x_mean"],
        stats["y_median"],
        lw=2.0,
        color=TOL_BLUE,
        zorder=3,
    )
    ax.axhline(0, color="black", lw=0.5, alpha=0.45)
    ax.set_xlabel("Router score")
    ax.set_ylabel("Expert neuron activation")
    ax.set_title("")
    ax.text(
        0.04,
        0.96,
        rf"median $\rho={median_rho:+.2f}$" + "\n" + rf"$p={format_p_value(p_value)}$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color="black",
    )
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_wiki_sequences(dataset_dir: str, max_seqs: int) -> np.ndarray:
    wiki_paths = [
        f"{dataset_dir}/wiki/part-0-00000.npy",
        f"{dataset_dir}/wiki/part-0-00001.npy",
    ]
    parts = [np.memmap(p, dtype=np.uint16, mode="r") for p in wiki_paths]
    tokens_flat = np.concatenate(parts)
    n_seqs = min(max_seqs, len(tokens_flat) // SEQ_LEN)
    return tokens_flat[: n_seqs * SEQ_LEN].reshape(n_seqs, SEQ_LEN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="moe-1b-269440-deepseek")
    parser.add_argument("--step", type=int, default=21000)
    parser.add_argument("--max-seqs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--name", default="router_score_expert_response_coupling")
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument(
        "--response",
        default="silu_mean",
        choices=["silu_mean", "preact_mean", "preact_max", "top10_preact"],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA/ROCm device, but torch.cuda.is_available() is false")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"device={device}")
    model, dataset_dir = build_model(args.run, args.step, device)
    h_cache, idx_cache, handles = register_router_hooks(model)
    seqs = load_wiki_sequences(dataset_dir, args.max_seqs)
    print(f"Processing {seqs.shape[0]} sequences x {SEQ_LEN} tokens = {seqs.size:,} tokens")

    logit_chunks: dict[int, list[np.ndarray]] = {l: [] for l in range(N_LAYERS)}
    response_chunks: dict[int, list[np.ndarray]] = {l: [] for l in range(N_LAYERS)}
    idx_chunks: dict[int, list[np.ndarray]] = {l: [] for l in range(N_LAYERS)}

    n_batches = (seqs.shape[0] + args.batch_size - 1) // args.batch_size
    try:
        with torch.inference_mode():
            for bi in range(n_batches):
                start = bi * args.batch_size
                end = min(start + args.batch_size, seqs.shape[0])
                batch = torch.from_numpy(seqs[start:end].astype(np.int64)).to(device)
                _ = model(batch)

                for layer in range(N_LAYERS):
                    h = h_cache[layer].float()
                    expert_idx = idx_cache[layer]
                    moe = model.blocks[str(layer)].feed_forward_moe
                    raw_logits = moe.router.get_expert_logits(h).float().gather(-1, expert_idx)
                    gate = gate_preacts_via_gmm(h, expert_idx, moe.experts.mlp).float()
                    if args.response == "silu_mean":
                        response = torch.nn.functional.silu(gate).mean(dim=-1)
                    elif args.response == "preact_mean":
                        response = gate.mean(dim=-1)
                    elif args.response == "preact_max":
                        response = gate.max(dim=-1).values
                    elif args.response == "top10_preact":
                        response = gate.topk(10, dim=-1).values.mean(dim=-1)
                    else:
                        raise NotImplementedError(args.response)

                    logit_chunks[layer].append(raw_logits.cpu().numpy().astype(np.float32))
                    response_chunks[layer].append(response.cpu().numpy().astype(np.float32))
                    idx_chunks[layer].append(expert_idx.cpu().numpy().astype(np.int16))

                if not args.quiet and (bi % 4 == 0 or bi == n_batches - 1):
                    print(f"  [{end}/{seqs.shape[0]} seqs]")
    finally:
        for handle in handles:
            handle.remove()

    all_x_z = []
    all_y_z = []
    all_group_rhos = []
    for layer in range(N_LAYERS):
        logits = np.concatenate(logit_chunks[layer], axis=0)
        response = np.concatenate(response_chunks[layer], axis=0)
        idx = np.concatenate(idx_chunks[layer], axis=0)
        x_z = zscore_by_expert(logits, idx)
        y_z = zscore_by_expert(response, idx)
        all_x_z.append(x_z.reshape(-1))
        all_y_z.append(y_z.reshape(-1))
        all_group_rhos.append(layer_expert_rhos(logits, response, idx))

    all_x = np.concatenate(all_x_z)
    all_y = np.concatenate(all_y_z)
    stats = binned_stats(all_x, all_y, args.bins)
    low, high = low_high_summary(all_x, all_y)
    group_rhos = np.concatenate(all_group_rhos)
    median_rho = float(np.median(group_rhos))
    positive_groups = int((group_rhos > 0).sum())
    sign_p = binomial_two_sided_p_value(positive_groups, len(group_rhos))
    plot(stats, outdir / args.name, median_rho=median_rho, p_value=sign_p)

    print("\nPaper numbers:")
    print(f"  layer-expert median Spearman rho: {median_rho:+.3f}")
    print(f"  positive layer-expert groups: {positive_groups}/{len(group_rhos)}")
    print(f"  sign-test p-value for positive correlations: {format_p_value(sign_p)}")
    print(f"  median response below vs above average router score: {low:+.3f}->{high:+.3f}")
    print(
        "  median normalized expert response, lowest->highest router-score group: "
        f"{stats['y_median'][0]:+.3f}->{stats['y_median'][-1]:+.3f}"
    )
    print(f"Wrote {outdir / (args.name + '.png')} and {outdir / (args.name + '.pdf')}")


if __name__ == "__main__":
    main()
