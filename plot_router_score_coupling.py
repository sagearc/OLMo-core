#!/usr/bin/env python3
"""
Plot router-score / expert-gate geometric coupling for the 21K DeepSeek MoE run.

The input is produced by ``collect_wiki_gate_acts.py`` and contains, for each
selected (token, rank) pair:

  - ``l{layer}_expert_score``: the model's actual post-gating, L1-normalized
    dispatch weight for the selected expert.
  - ``l{layer}_gate_preact``: the selected expert's pre-SiLU gate vector
    ``h W_gate^T``.
  - ``l{layer}_expert_idx``: the selected expert id.

For each layer we compute the L2 norm of the gate vector, z-score it per expert,
then bin dispatch weights into quantiles. This gives the compact main-text plot:
continuous router preference predicts the selected expert's internal response.

Generates:
  router_score_gate_coupling.png
  router_score_gate_coupling.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

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

N_LAYERS = 9
N_EXPERTS = 64
N_BINS = 10

TOL_BLUE = "#4477AA"
TOL_RED = "#CC6677"
TOL_GREY = "#BBBBBB"


def per_expert_z_gate_norm_cpu(gate_fp16: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Return z-scored gate-vector norms with normalization done per expert."""
    norms = np.linalg.norm(gate_fp16, axis=2).astype(np.float64)
    flat_idx = idx.ravel().astype(np.intp)
    flat_norm = norms.ravel()

    counts = np.bincount(flat_idx, minlength=N_EXPERTS)
    sums = np.bincount(flat_idx, weights=flat_norm, minlength=N_EXPERTS)
    sums_sq = np.bincount(flat_idx, weights=flat_norm * flat_norm, minlength=N_EXPERTS)
    if counts.min() == 0:
        dead = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"experts with zero selections: {dead}")

    mu = sums / counts
    var = np.maximum(sums_sq / counts - mu * mu, 1e-8)
    sigma = np.sqrt(var)
    return ((norms - mu[idx]) / sigma[idx]).astype(np.float32)


def per_expert_z_gate_norm(gate_fp16: np.ndarray, idx: np.ndarray, device: torch.device) -> np.ndarray:
    """Return z-scored gate-vector norms with normalization done per expert."""
    if device.type == "cpu":
        return per_expert_z_gate_norm_cpu(gate_fp16, idx)

    with torch.inference_mode():
        gate = torch.as_tensor(gate_fp16, device=device)
        idx_t = torch.as_tensor(idx, device=device, dtype=torch.long)

        norms = torch.linalg.vector_norm(gate.float(), dim=2)
        flat_idx = idx_t.reshape(-1)
        flat_norm = norms.reshape(-1)

        counts = torch.bincount(flat_idx, minlength=N_EXPERTS).float()
        if bool((counts == 0).any().item()):
            dead = torch.nonzero(counts == 0, as_tuple=False).flatten().cpu().tolist()
            raise ValueError(f"experts with zero selections: {dead}")

        sums = torch.bincount(flat_idx, weights=flat_norm, minlength=N_EXPERTS)
        sums_sq = torch.bincount(flat_idx, weights=flat_norm * flat_norm, minlength=N_EXPERTS)
        mu = sums / counts
        var = torch.clamp(sums_sq / counts - mu * mu, min=1e-8)
        sigma = torch.sqrt(var)
        z = ((norms - mu[idx_t]) / sigma[idx_t]).cpu().numpy().astype(np.float32)

    del gate, idx_t, norms, flat_idx, flat_norm, counts, sums, sums_sq, mu, var, sigma
    torch.cuda.empty_cache()
    return z


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average-tie ranks, 1-indexed, without requiring scipy."""
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
    """Spearman rho without scipy, ignoring non-finite pairs."""
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


def binned_stats(scores: np.ndarray, z_gate: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Quantile-bin scores and summarize gate response in each bin."""
    flat_scores = scores.reshape(-1).astype(np.float64)
    flat_z = z_gate.reshape(-1).astype(np.float64)
    mask = np.isfinite(flat_scores) & np.isfinite(flat_z)
    flat_scores = flat_scores[mask]
    flat_z = flat_z[mask]

    edges = np.quantile(flat_scores, np.linspace(0, 1, n_bins + 1))
    bins = np.searchsorted(edges[1:-1], flat_scores, side="right")

    x_mean = np.empty(n_bins, dtype=np.float64)
    y_mean = np.empty(n_bins, dtype=np.float64)
    y_median = np.empty(n_bins, dtype=np.float64)
    y_p25 = np.empty(n_bins, dtype=np.float64)
    y_p75 = np.empty(n_bins, dtype=np.float64)
    counts = np.empty(n_bins, dtype=np.int64)

    for b in range(n_bins):
        sel = bins == b
        counts[b] = int(sel.sum())
        x_mean[b] = flat_scores[sel].mean()
        y_mean[b] = flat_z[sel].mean()
        y_median[b] = np.median(flat_z[sel])
        y_p25[b], y_p75[b] = np.percentile(flat_z[sel], [25, 75])

    return {
        "x_mean": x_mean,
        "y_mean": y_mean,
        "y_median": y_median,
        "y_p25": y_p25,
        "y_p75": y_p75,
        "counts": counts,
        "rho": np.array(spearmanr(flat_scores, flat_z), dtype=np.float64),
    }


def layer_expert_rhos(scores: np.ndarray, z_gate: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Compute rho inside each selected (layer, expert) group."""
    flat_scores = scores.reshape(-1).astype(np.float64)
    flat_z = z_gate.reshape(-1).astype(np.float64)
    flat_idx = idx.reshape(-1).astype(np.intp)
    rhos = []
    for expert in range(N_EXPERTS):
        sel = flat_idx == expert
        if sel.sum() >= 128:
            rho = spearmanr(flat_scores[sel], flat_z[sel])
            if np.isfinite(rho):
                rhos.append(rho)
    return np.array(rhos, dtype=np.float64)


def plot(global_stats: dict[str, np.ndarray], per_layer: dict[int, dict[str, np.ndarray]], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.35))

    for layer in range(N_LAYERS):
        st = per_layer[layer]
        ax.plot(
            st["x_mean"],
            st["y_median"],
            color=TOL_GREY,
            lw=0.7,
            alpha=0.55,
            zorder=1,
        )

    ax.fill_between(
        global_stats["x_mean"],
        global_stats["y_p25"],
        global_stats["y_p75"],
        color=TOL_BLUE,
        alpha=0.16,
        linewidth=0,
        label="25-75%",
        zorder=2,
    )
    ax.plot(
        global_stats["x_mean"],
        global_stats["y_median"],
        marker="o",
        ms=3.8,
        lw=1.8,
        color=TOL_BLUE,
        label="Median",
        zorder=3,
    )
    ax.plot(
        global_stats["x_mean"],
        global_stats["y_mean"],
        marker="D",
        ms=3.0,
        lw=1.1,
        color=TOL_RED,
        label="Mean",
        zorder=4,
    )

    ax.axhline(0, color="black", lw=0.5, alpha=0.45)
    ax.set_xlabel("Selected-expert routing weight (decile mean)")
    ax.set_ylabel("Normalized gate response")
    ax.set_title("Router score predicts expert response")
    ax.legend(frameon=False, fontsize=7, loc="upper left", handlelength=1.3)
    sns.despine(ax=ax)
    fig.tight_layout(pad=0.4)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="wiki_gate_acts_deepseek_step21k.npz")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--name", default="router_score_gate_coupling")
    parser.add_argument("--bins", type=int, default=N_BINS)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for gate-norm/zscore computation: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    data = np.load(args.input)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested CUDA/ROCm device, but torch.cuda.is_available() is false")
    print(f"Using {device} for gate-norm/zscore computation")

    score_chunks = []
    z_chunks = []
    per_layer: dict[int, dict[str, np.ndarray]] = {}
    all_group_rhos = []

    for layer in range(N_LAYERS):
        gate = data[f"l{layer}_gate_preact"]
        idx = data[f"l{layer}_expert_idx"].astype(np.intp)
        scores = data[f"l{layer}_expert_score"].astype(np.float32)
        z_gate = per_expert_z_gate_norm(gate, idx, device)

        per_layer[layer] = binned_stats(scores, z_gate, args.bins)
        group_rhos = layer_expert_rhos(scores, z_gate, idx)
        all_group_rhos.append(group_rhos)

        score_chunks.append(scores.reshape(-1))
        z_chunks.append(z_gate.reshape(-1))

        if not args.quiet:
            st = per_layer[layer]
            print(
                f"  L{layer}: rho={float(st['rho']):+.3f}, "
                f"median first->last bin={st['y_median'][0]:+.3f}->{st['y_median'][-1]:+.3f}"
            )

    all_scores = np.concatenate(score_chunks)
    all_z = np.concatenate(z_chunks)
    global_stats = binned_stats(all_scores, all_z, args.bins)
    group_rhos = np.concatenate(all_group_rhos)

    plot(global_stats, per_layer, outdir / args.name)

    print("\nPaper numbers:")
    print(f"  pooled Spearman rho: {float(global_stats['rho']):+.3f}")
    print(f"  layer-expert median rho: {np.median(group_rhos):+.3f}")
    print(f"  positive layer-expert groups: {(group_rhos > 0).sum()}/{len(group_rhos)}")
    print(
        "  median normalized gate response, lowest->highest score decile: "
        f"{global_stats['y_median'][0]:+.3f}->{global_stats['y_median'][-1]:+.3f}"
    )
    print(f"Wrote {outdir / (args.name + '.png')} and {outdir / (args.name + '.pdf')}")


if __name__ == "__main__":
    main()
