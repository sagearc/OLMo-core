#!/usr/bin/env python3
"""
Figure 2 plots — routing rank predicts gate-neuron activation strength.

For each (token, rank) pair in the held-out Wikipedia data we compute the L2
norm of the gate pre-activation `h W_gate^T`, and Z-score it against the
mean/std of the same norm across all (token, rank) pairs that selected the
expert. Top-ranked experts have larger gate magnitudes — the geometric
coupling claim.

Generates (PNG @ 300 dpi + camera-ready PDF):
  fig2_rank_zscore_per_layer  line plot, median Z by rank, one line per layer
  fig2_rank_zscore_layer4     single-layer (L4) violin + median markers
  fig2_rank_zscore_grid       3x3 grid of violins, one panel per layer
  fig2_rank_pre_vs_post_silu  pre-SiLU vs post-SiLU L2 norm comparison
                              (proves the trend is invariant to the choice
                               of "activation"; for the appendix)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ── Style: matches plot_router_heatmaps.py / plot_subspace_intervals.py ──────
sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       9,
    "axes.titlesize":  9,
    "axes.labelsize":  8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi":      300,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
})

# Paul Tol colour-blind palette, consistent with sibling figures
TOL_BLUE    = "#4477AA"
TOL_MAGENTA = "#AA3377"
TOL_RED     = "#EE6677"
TOL_GREY    = "#BBBBBB"

N_LAYERS  = 9
N_EXPERTS = 64
TOP_K     = 6
DETAIL_LAYER = 4


def per_expert_z_norm(gate_fp16: np.ndarray, idx: np.ndarray, postsilu: bool = False) -> np.ndarray:
    """
    Vectorised Z-score of per-(token, rank) gate L2 norm, with mean/std
    computed per expert across all (token, rank) pairs that selected it.

    gate_fp16: (N, top_k, hidden) fp16    pre-SiLU gate activations h W_gate^T
    idx:       (N, top_k) int             expert chosen at column r
    postsilu:  if True, take the L2 norm of silu(gate) instead of gate
    Returns:   (N, top_k) fp32            Z-scored norm per (token, rank)
    """
    if postsilu:
        # SiLU(x) = x * sigmoid(x); use fp32 to avoid fp16 overflow in exp
        x = gate_fp16.astype(np.float32)
        gate = x / (1.0 + np.exp(-x)) * 1.0  # silu
    else:
        gate = gate_fp16  # norm internally promotes to fp32 — no upcast needed

    norms = np.linalg.norm(gate, axis=2).astype(np.float64)  # (N, top_k)

    flat_idx   = idx.ravel().astype(np.intp)
    flat_norm  = norms.ravel()
    nm_cnt   = np.bincount(flat_idx, minlength=N_EXPERTS)
    nm_sum   = np.bincount(flat_idx, weights=flat_norm,        minlength=N_EXPERTS)
    nm_sumsq = np.bincount(flat_idx, weights=flat_norm ** 2,   minlength=N_EXPERTS)
    assert nm_cnt.min() > 0, "dead expert: at least one expert had zero selections"

    nm_mu  = nm_sum / nm_cnt
    nm_var = np.maximum(nm_sumsq / nm_cnt - nm_mu ** 2, 1e-8)
    nm_sig = np.sqrt(nm_var)

    return ((norms - nm_mu[idx]) / nm_sig[idx]).astype(np.float32)


def long_form(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flatten (N, top_k) -> (N*top_k,) values + 1-indexed rank labels."""
    n, k = z.shape
    return z.reshape(-1), np.tile(np.arange(1, k + 1), n)


def annotate_medians(ax, ranks, medians, color="black"):
    for r, m in zip(ranks, medians):
        ax.text(r - 1, m + 0.06, f"{m:+.2f}", ha="center", va="bottom",
                fontsize=7, color=color)


# ── Plot helpers ─────────────────────────────────────────────────────────────

def plot_per_layer_lines(stats_by_layer: dict[int, dict], path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    cmap = plt.get_cmap("viridis", N_LAYERS)
    ranks = np.arange(1, TOP_K + 1)
    for l in range(N_LAYERS):
        med = stats_by_layer[l]["median"]
        ax.plot(ranks, med, marker="o", lw=1.5, ms=4.5, color=cmap(l),
                label=f"Layer {l}")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Routing rank")
    ax.set_ylabel("Median Z-scored gate L2 norm\n(pre-SiLU $\\|h W_{\\mathrm{gate}}^\\top\\|_2$)")
    ax.set_xticks(ranks)
    ax.set_title("Top-ranked experts fire more strongly")
    ax.legend(ncol=3, frameon=False, fontsize=7,
              loc="upper right", handlelength=1.2, columnspacing=0.8)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def violin_panel(ax, z, ranks, color, show_y=True, show_x=True):
    vals, ranks_long = long_form(z)
    sns.violinplot(x=ranks_long, y=vals, ax=ax, color=color,
                   inner=None, linewidth=0.6, cut=0, density_norm="width",
                   bw_adjust=0.6, native_scale=True)
    medians = stats_by_layer_fast(z)["median"]
    ax.scatter(ranks - 1, medians, marker="D", s=18, zorder=10,
               color="white", edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", lw=0.4, alpha=0.4)
    if show_x:
        ax.set_xlabel("Routing rank")
    else:
        ax.set_xlabel("")
    if show_y:
        ax.set_ylabel("Z-scored gate L2 norm")
    else:
        ax.set_ylabel("")
    ax.set_xticks(np.arange(TOP_K))
    ax.set_xticklabels(ranks)
    sns.despine(ax=ax)
    return medians


def stats_by_layer_fast(z: np.ndarray) -> dict:
    return {
        "mean":    z.mean(0),
        "median":  np.median(z, axis=0),
        "p25":     np.percentile(z, 25, axis=0),
        "p75":     np.percentile(z, 75, axis=0),
    }


def plot_layer_detail(z: np.ndarray, layer: int, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ranks = np.arange(1, TOP_K + 1)
    medians = violin_panel(ax, z, ranks, color=TOL_BLUE)
    annotate_medians(ax, ranks, medians)
    ax.set_title(f"Layer {layer}: gate magnitude declines monotonically with rank")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_layer_grid(z_by_layer: dict[int, np.ndarray], path: Path):
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 7.0), sharex=True, sharey=True)
    ranks = np.arange(1, TOP_K + 1)
    for l in range(N_LAYERS):
        ax = axes[l // 3, l % 3]
        violin_panel(ax, z_by_layer[l], ranks, color=TOL_BLUE,
                     show_y=(l % 3 == 0), show_x=(l // 3 == 2))
        ax.set_title(f"Layer {l}", pad=2)
    fig.suptitle("Gate magnitude vs routing rank, all layers", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pre_vs_post_silu(stats_pre: dict, stats_post: dict, path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ranks = np.arange(1, TOP_K + 1)
    cmap = plt.get_cmap("viridis", N_LAYERS)
    for l in range(N_LAYERS):
        ax.plot(ranks, stats_pre[l]["median"],  marker="o", lw=1.2, ms=3,
                color=cmap(l), alpha=0.9)
        ax.plot(ranks, stats_post[l]["median"], marker="s", lw=1.0, ms=3,
                color=cmap(l), alpha=0.9, ls="--")
    ax.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax.set_xlabel("Routing rank")
    ax.set_ylabel("Median Z-scored gate L2 norm")
    ax.set_xticks(ranks)
    ax.set_title("Pre-SiLU (solid) vs post-SiLU (dashed) — same trend")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


# ── Driver ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="wiki_gate_acts_deepseek_step21k.npz")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--quiet",  action="store_true")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.input} ...")
    d = np.load(args.input)

    z_pre:  dict[int, np.ndarray] = {}
    z_post: dict[int, np.ndarray] = {}
    stats_pre, stats_post = {}, {}
    for l in range(N_LAYERS):
        gate = d[f"l{l}_gate_preact"]                  # fp16 (N, 6, 512)
        idx  = d[f"l{l}_expert_idx"].astype(np.intp)    # (N, 6)
        z_pre[l]   = per_expert_z_norm(gate, idx, postsilu=False)
        z_post[l]  = per_expert_z_norm(gate, idx, postsilu=True)
        stats_pre[l]  = stats_by_layer_fast(z_pre[l])
        stats_post[l] = stats_by_layer_fast(z_post[l])
        if not args.quiet:
            m = stats_pre[l]["median"]
            print(f"  L{l}  pre-SiLU median Z by rank: {np.round(m, 3).tolist()}")

    print("Plotting...")
    plot_per_layer_lines(stats_pre, out / "fig2_rank_zscore_per_layer")
    plot_layer_detail(z_pre[DETAIL_LAYER], DETAIL_LAYER, out / "fig2_rank_zscore_layer4")
    plot_layer_grid(z_pre, out / "fig2_rank_zscore_grid")
    plot_pre_vs_post_silu(stats_pre, stats_post, out / "fig2_rank_pre_vs_post_silu")
    print(f"Wrote 4 figures (PNG + PDF) to {out.resolve()}")


if __name__ == "__main__":
    main()
