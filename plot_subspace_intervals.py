"""
Router-expert subspace alignment following SD-MoE (arxiv:2602.12556).

Score(i, j, L) = sum_{m=1}^{k} cos^2(r_i, v_m(j,L))
              = ||V_k(j,L) @ r_i||^2

where r_i is the unit-norm router vector for expert i at layer L,
V_k(j,L) are the top-k right singular vectors of expert j's w1 (shape k x D),
and k = ceil(1% * min(H, D)) = 6.

Because r_i is a unit vector and V_k rows are orthonormal, Bessel's inequality
gives 0 <= Score <= 1. Random-unit-vector baseline: E[Score] = k/D ~ 0.0059.

Key result: specificity = Score(i,i) / mean_{j!=i} Score(i,j)
  > 1: router i aligns more with its own expert than with others (co-specialisation)
  < 1: no specificity (routing collapse)
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent / "src"))
from olmo_core.distributed.checkpoint import load_keys

N_LAYERS, N_EXPERTS, D_MODEL, H_EXPERT = 9, 64, 1024, 512
K_SVD      = int(np.ceil(0.01 * min(H_EXPERT, D_MODEL)))   # = 6
RANDOM_BL  = K_SVD / D_MODEL                                # = 6/1024 ~ 0.00586
SEL_LAYERS = [0, N_LAYERS // 2, N_LAYERS - 1]              # = [0, 4, 8]

CKPTS = {
    "With auxiliary loss": "runs/moe-1b-269439-baseline/step21000/model_and_optim",
    "Without auxiliary loss": "runs/moe-1b-269440-deepseek/step21000/model_and_optim",
}
COLORS = {"With auxiliary loss": "#4477AA", "Without auxiliary loss": "#AA3377"}

# ── Load ──────────────────────────────────────────────────────────────────────

def load_router(ckpt):
    """(N_LAYERS, N_EXPERTS, D_MODEL), rows exactly unit-norm."""
    keys = [f"model.blocks.{L}.feed_forward_moe.router.weight" for L in range(N_LAYERS)]
    raw = np.stack([w.reshape(N_EXPERTS, D_MODEL).float().numpy()
                    for w in load_keys(ckpt, keys)])
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    assert (norms > 1e-6).all(), "zero-norm router vector found"
    return raw / norms


def load_w1(ckpt):
    """(N_LAYERS, N_EXPERTS, H_EXPERT, D_MODEL)."""
    keys = [f"model.blocks.{L}.feed_forward_moe.experts.mlp.w1" for L in range(N_LAYERS)]
    return np.stack([w.reshape(N_EXPERTS, H_EXPERT, D_MODEL).float().numpy()
                     for w in load_keys(ckpt, keys)])


print("Loading checkpoints...")
router = {n: load_router(p) for n, p in CKPTS.items()}
expert = {n: load_w1(p)     for n, p in CKPTS.items()}
print("  done.")

# ── Alignment ─────────────────────────────────────────────────────────────────

def full_alignment_matrix(W_all, R_all):
    """
    Returns (N_LAYERS, N_EXPERTS, N_EXPERTS) float32.
    scores[L, i, j] = sum_{m=1}^{k} cos^2(r_i, v_m(j)) in [0, 1].
    """
    scores = np.zeros((N_LAYERS, N_EXPERTS, N_EXPERTS), np.float32)
    for L in range(N_LAYERS):
        _, _, Vt = np.linalg.svd(W_all[L], full_matrices=False)  # (E, min(H,D), D)
        Vk   = Vt[:, :K_SVD, :]                                   # (E, K, D)
        proj = np.einsum("jmd,id->jmi", Vk, R_all[L])             # (E_exp, K, E_rtr)
        scores[L] = (proj ** 2).sum(axis=1).T                     # (E_rtr, E_exp)
    assert scores.max() <= 1.0 + 1e-5, f"score out of [0,1]: {scores.max()}"
    return scores


print(f"Computing alignment matrices (k={K_SVD})...")
full = {n: full_alignment_matrix(expert[n], router[n]) for n in CKPTS}
print("  done.")

# ── Derived metrics ───────────────────────────────────────────────────────────

def matched_mismatched(mat):
    """mat (L, E, E) → matched (L, E), mean_mismatched (L, E)."""
    matched    = np.diagonal(mat, axis1=1, axis2=2).copy()        # (L, E)
    mismatched = (mat.sum(axis=2) - matched) / (N_EXPERTS - 1)    # (L, E)
    return matched, mismatched


_mm         = {n: matched_mismatched(full[n]) for n in CKPTS}
matched     = {n: _mm[n][0] for n in CKPTS}
mismatched  = {n: _mm[n][1] for n in CKPTS}
specificity = {n: matched[n] / (mismatched[n] + 1e-8) for n in CKPTS}

layers = np.arange(N_LAYERS)

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({"font.family": "serif", "font.size": 9,
                     "axes.titlesize": 9, "axes.labelsize": 8,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 300})

# ── Figure 1: alignment heatmaps (64×64) ──────────────────────────────────────
vmax   = max(full[n].max() for n in CKPTS)
cmap   = "YlOrRd"
norm_h = Normalize(vmin=0, vmax=vmax)

n_rows, n_cols = len(CKPTS), len(SEL_LAYERS)
fig = plt.figure(figsize=(6.5, 4.6))
gs  = mgridspec.GridSpec(n_rows, n_cols + 1, figure=fig,
                         wspace=0.08, hspace=0.12,
                         width_ratios=[1, 1, 1, 0.06])

for row, model_name in enumerate(CKPTS):
    for col, L in enumerate(SEL_LAYERS):
        ax = fig.add_subplot(gs[row, col])
        sns.heatmap(full[model_name][L], ax=ax, cmap=cmap, norm=norm_h,
                    square=True, cbar=False,
                    xticklabels=False, yticklabels=False,
                    linewidths=0, rasterized=True)

        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_linewidth(0.5); spine.set_edgecolor("#555555")

        label = chr(ord("a") + row * n_cols + col)
        title = f"({label}) Layer {L}" if row == 0 else f"({label})"
        ax.set_title(title, fontsize=8.5, pad=3, loc="left", color="black")

        if col == 0:
            ax.set_ylabel(model_name, fontsize=9, labelpad=4, rotation=90, va="center")

        spec = specificity[model_name][L].mean()
        ax.text(0.5, -0.04, f"spec = {spec:.2f}×",
                transform=ax.transAxes, fontsize=7, va="top", ha="center", color="black")

cbar_ax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(ScalarMappable(norm=norm_h, cmap=cmap), cax=cbar_ax)
cb.set_label(r"$\sum_m \cos^2(\hat{r}_i,\,v_m^{(j)})$", fontsize=7, labelpad=4)
cb.ax.tick_params(labelsize=7)

fig.savefig("router_heatmaps_neurips.pdf", bbox_inches="tight")
fig.savefig("router_heatmaps_neurips.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Saved router_heatmaps_neurips.pdf / .png")

# ── Figure 2: matched / mismatched / specificity per layer ────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

ax = axes[0]
ax.axhline(RANDOM_BL, color="gray", ls="--", lw=1, label=f"random ({RANDOM_BL:.4f})")
for name, col in COLORS.items():
    ax.plot(layers, matched[name].mean(axis=1), marker="o", color=col, lw=1.8, label=name)
ax.set_xlabel("MoE layer")
ax.set_ylabel(r"$\sum_m \cos^2(\hat{r}_i,\, v_m^{(i)})$")
ax.set_title("Matched: router $i$ vs. own expert $i$")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(RANDOM_BL, color="gray", ls="--", lw=1, label=f"random ({RANDOM_BL:.4f})")
for name, col in COLORS.items():
    ax.plot(layers, mismatched[name].mean(axis=1), marker="o", color=col, lw=1.8, label=name)
ax.set_xlabel("MoE layer")
ax.set_ylabel(r"$\sum_m \cos^2(\hat{r}_i,\, v_m^{(j)})$  (mean $j\neq i$)")
ax.set_title("Mismatched: router $i$ vs. other experts $j\\neq i$")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
for name, col in COLORS.items():
    mu  = specificity[name].mean(axis=1)
    std = specificity[name].std(axis=1)
    ax.plot(layers, mu, marker="o", color=col, lw=1.8, label=name)
    ax.fill_between(layers, mu - std, mu + std, alpha=0.15, color=col)
ax.axhline(1.0, color="gray", ls="--", lw=1, label="no specificity (1×)")
ax.set_xlabel("MoE layer")
ax.set_ylabel("matched / mean-mismatched")
ax.set_title("Specificity (matched / mismatched)\nhigher = router co-specialised with own expert")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("subspace_intervals_compare.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved subspace_intervals_compare.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nk={K_SVD}  random baseline={RANDOM_BL:.5f}  (= k/D = {K_SVD}/{D_MODEL})")
print(f"{'Model':<20s}  {'matched':>9s}  {'mismatch':>9s}  {'specificity':>12s}")
print("-" * 60)
for n in CKPTS:
    print(f"{n:<20s}  {matched[n].mean():9.5f}  {mismatched[n].mean():9.5f}  "
          f"{specificity[n].mean():12.2f}x")
