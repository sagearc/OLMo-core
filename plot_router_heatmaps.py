"""
Publication-quality router cosine-similarity heatmaps for NeurIPS 2026.
Both models at the same step (27 500).
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent / "src"))
from olmo_core.distributed.checkpoint import load_keys

# ── Config ────────────────────────────────────────────────────────────────────
N_LAYERS, N_EXPERTS, D_MODEL = 9, 64, 1024
SEL_LAYERS = [0, 4, 8]
CKPTS = {
    "With auxiliary loss": "runs/moe-1b-269439-baseline/step21000/model_and_optim",
    "Without auxiliary loss": "runs/moe-1b-269440-deepseek/step21000/model_and_optim",
}
VMIN, VMAX, VCENTER = -0.3, 1.0, 0.0

# ── Style ─────────────────────────────────────────────────────────────────────
sns.set_theme(style="white", font_scale=1.0)
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        9,
    "axes.titlesize":   9,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "figure.dpi":       300,
})

# ── Load ──────────────────────────────────────────────────────────────────────
def load_router(ckpt: str) -> np.ndarray:
    keys = [f"model.blocks.{L}.feed_forward_moe.router.weight" for L in range(N_LAYERS)]
    raw = np.stack([w.reshape(N_EXPERTS, D_MODEL).float().numpy()
                    for w in load_keys(ckpt, keys)])
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    return raw / (norms + 1e-8)

print("Loading checkpoints...")
router = {n: load_router(p) for n, p in CKPTS.items()}
print("  done.")

# ── Layout ────────────────────────────────────────────────────────────────────
# 2 rows (models) × 3 cols (layers) + narrow colorbar column
n_rows, n_cols = len(CKPTS), len(SEL_LAYERS)
fig = plt.figure(figsize=(6.5, 4.6))

# GridSpec: 3 heatmap cols + 1 thin colorbar col
from matplotlib.gridspec import GridSpec
gs = GridSpec(n_rows, n_cols + 1,
              figure=fig,
              wspace=0.08, hspace=0.12,
              width_ratios=[1, 1, 1, 0.06])

cmap = sns.diverging_palette(220, 20, as_cmap=True)   # blue–white–red
norm = Normalize(vmin=VMIN, vmax=VMAX)

for row, (model_name, Wn_all) in enumerate(
        zip(CKPTS.keys(), [router[n] for n in CKPTS])):

    for col, L in enumerate(SEL_LAYERS):
        ax = fig.add_subplot(gs[row, col])

        S = Wn_all[L] @ Wn_all[L].T          # (64, 64) cosine sim matrix
        sns.heatmap(
            S, ax=ax,
            cmap=cmap, norm=norm,
            square=True, cbar=False,
            xticklabels=False, yticklabels=False,
            linewidths=0, rasterized=True,
        )

        # Thin border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#555555")

        # Panel letter + layer — above the axes, never inside the heatmap
        label = chr(ord("a") + row * n_cols + col)
        title = f"({label}) Layer {L}" if row == 0 else f"({label})"
        ax.set_title(title, fontsize=8.5, pad=3, loc="left", color="black")

        # Row label (model name) on first column only
        if col == 0:
            ax.set_ylabel(model_name, fontsize=9, labelpad=4, rotation=90, va="center")

        # Mean off-diagonal cosine — below the axes, centred, always black
        off = S.copy(); np.fill_diagonal(off, np.nan)
        mean_cos = np.nanmean(off)
        ax.text(0.5, -0.04, f"μ = {mean_cos:.2f}",
                transform=ax.transAxes,
                fontsize=7, va="top", ha="center", color="black")

# Shared colorbar
cbar_ax = fig.add_subplot(gs[:, -1])
cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap),
                  cax=cbar_ax, orientation="vertical")
cb.set_label("cosine similarity", fontsize=8, labelpad=4)
cb.ax.tick_params(labelsize=7)
cb.set_ticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

fig.savefig("router_heatmaps_neurips.pdf",  bbox_inches="tight")
fig.savefig("router_heatmaps_neurips.png",  bbox_inches="tight", dpi=300)
print("Saved router_heatmaps_neurips.pdf / .png")
