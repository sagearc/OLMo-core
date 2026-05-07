"""
Compare MoE router weight vectors between:
  - runs/moe-1b-269439-baseline/step27500   (baseline, final)
  - runs/moe-1b-269440-deepseek/step27500   (deepseek, same step)
  - runs/moe-1b-269440-deepseek/step39750   (deepseek, final)

Metrics per layer:
  1. Mean pairwise cosine similarity (lower = more diverse directions)
  2. Nearest-neighbour cosine similarity distribution
  3. Effective rank of the router matrix (higher = more isotropic)
  4. Pairwise cosine sim heatmap (64x64) for a selected layer
  5. Score-bias distribution (DeepSeek only) — proxy for load balance
  6. Cross-model alignment: how well do deepseek's router dirs match baseline's?
"""
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, str(Path(__file__).parent / "src"))
from olmo_core.distributed.checkpoint import load_keys

N_LAYERS = 9
N_EXPERTS = 64
D_MODEL = 1024

CKPTS = {
    "baseline@27k":  "runs/moe-1b-269439-baseline/step27500/model_and_optim",
    "deepseek@27k":  "runs/moe-1b-269440-deepseek/step27500/model_and_optim",
    "deepseek@39k":  "runs/moe-1b-269440-deepseek/step39750/model_and_optim",
}
COLORS = {
    "baseline@27k": "#4477AA",
    "deepseek@27k": "#EE6677",
    "deepseek@39k": "#AA3377",
}

# ──────────────────────────────────────────────────────────────────────────────
# Load
# ──────────────────────────────────────────────────────────────────────────────

def load_router_weights(ckpt_dir: str) -> np.ndarray:
    """Returns (N_LAYERS, N_EXPERTS, D_MODEL) float32."""
    keys = [f"model.blocks.{L}.feed_forward_moe.router.weight" for L in range(N_LAYERS)]
    weights = []
    for w in load_keys(ckpt_dir, keys):
        weights.append(w.reshape(N_EXPERTS, D_MODEL).float().numpy())
    return np.stack(weights)   # (L, E, D)


def load_score_bias(ckpt_dir: str) -> np.ndarray:
    """Returns (N_LAYERS, N_EXPERTS) float32, or None if not present."""
    keys = [f"model.blocks.{L}.feed_forward_moe.router.score_bias" for L in range(N_LAYERS)]
    try:
        biases = [b.float().numpy() for b in load_keys(ckpt_dir, keys)]
        return np.stack(biases)
    except Exception:
        return None


print("Loading checkpoints...")
router = {name: load_router_weights(path) for name, path in CKPTS.items()}
score_bias = {name: load_score_bias(path) for name, path in CKPTS.items()}
print("  done.")

# ──────────────────────────────────────────────────────────────────────────────
# Per-layer metrics
# ──────────────────────────────────────────────────────────────────────────────

def cosine_sim_matrix(W: np.ndarray) -> np.ndarray:
    """W: (E, D) → sim matrix (E, E)."""
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W_n = W / (norms + 1e-8)
    return W_n @ W_n.T


def mean_pairwise_cosine(W: np.ndarray) -> float:
    """Mean cosine sim over all i≠j pairs."""
    S = cosine_sim_matrix(W)
    E = W.shape[0]
    return (S.sum() - np.trace(S)) / (E * (E - 1))


def nn_cosine(W: np.ndarray) -> np.ndarray:
    """For each expert, max cosine sim to any other expert. Returns (E,)."""
    S = cosine_sim_matrix(W)
    np.fill_diagonal(S, -1.0)
    return S.max(axis=1)


def effective_rank(W: np.ndarray) -> float:
    """Effective rank = exp(entropy of normalised squared singular values)."""
    sv = np.linalg.svd(W, compute_uv=False)
    p = sv**2 / (sv**2).sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


metrics = {}
for name, W_all in router.items():
    mpcs, nn_all, er = [], [], []
    for L in range(N_LAYERS):
        W = W_all[L]
        mpcs.append(mean_pairwise_cosine(W))
        nn_all.append(nn_cosine(W))
        er.append(effective_rank(W))
    metrics[name] = {
        "mean_pairwise_cosine": np.array(mpcs),
        "nn_cosine": np.array(nn_all),
        "effective_rank": np.array(er),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Cross-model alignment (baseline vs deepseek@39k)
# ──────────────────────────────────────────────────────────────────────────────

def best_match_cosine(Wa: np.ndarray, Wb: np.ndarray) -> np.ndarray:
    """For each expert in Wa, max cosine sim to any expert in Wb. Returns (E,)."""
    Sa = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-8)
    Sb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-8)
    return (Sa @ Sb.T).max(axis=1)   # (E,)


cross_match = {}   # (name_a, name_b) → (L, E) array
for (na, nb) in [("baseline@27k", "deepseek@39k"),
                 ("baseline@27k", "deepseek@27k")]:
    arr = np.stack([best_match_cosine(router[na][L], router[nb][L])
                    for L in range(N_LAYERS)])
    cross_match[(na, nb)] = arr   # (L, E)

# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: global metrics across layers
# ──────────────────────────────────────────────────────────────────────────────

layers = np.arange(N_LAYERS)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
for name in CKPTS:
    ax.plot(layers, metrics[name]["mean_pairwise_cosine"],
            marker="o", color=COLORS[name], label=name)
ax.set_xlabel("MoE layer"); ax.set_ylabel("mean pairwise cosine sim")
ax.set_title("Router diversity\n(lower = more diverse)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for name in CKPTS:
    vals = metrics[name]["nn_cosine"]
    for L in range(N_LAYERS):
        ax.scatter([L] * N_EXPERTS, vals[L],
                   alpha=0.25, s=8, color=COLORS[name],
                   label=name if L == 0 else None)
    ax.plot(layers, vals.mean(axis=1), marker="o", color=COLORS[name], lw=2)
ax.set_xlabel("MoE layer"); ax.set_ylabel("NN cosine sim (per expert)")
ax.set_title("Nearest-neighbour cosine sim\n(lower = more separated)")
ax.legend(fontsize=8, markerscale=3); ax.grid(alpha=0.3)

ax = axes[2]
for name in CKPTS:
    ax.plot(layers, metrics[name]["effective_rank"],
            marker="o", color=COLORS[name], label=name)
ax.set_xlabel("MoE layer"); ax.set_ylabel("effective rank")
ax.set_title("Router effective rank\n(higher = more isotropic)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("router_global_metrics.png", dpi=150)
plt.close(fig)
print("Saved router_global_metrics.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: pairwise cosine sim heatmaps for 3 selected layers
# ──────────────────────────────────────────────────────────────────────────────

SEL_LAYERS = [0, 4, 8]
SHOW_MODELS = ["baseline@27k", "deepseek@39k"]
fig, axes = plt.subplots(len(SEL_LAYERS), len(SHOW_MODELS),
                          figsize=(len(SHOW_MODELS) * 5, len(SEL_LAYERS) * 4.5))

for row, L in enumerate(SEL_LAYERS):
    for col, name in enumerate(SHOW_MODELS):
        S = cosine_sim_matrix(router[name][L])
        im = axes[row, col].imshow(S, vmin=-0.5, vmax=1.0, cmap="RdBu_r",
                                    aspect="auto", interpolation="nearest")
        axes[row, col].set_title(f"{name}  layer {L}")
        axes[row, col].set_xlabel("expert"); axes[row, col].set_ylabel("expert")
        plt.colorbar(im, ax=axes[row, col])

fig.suptitle("Pairwise cosine similarity of router weight vectors (64×64)", y=1.01)
fig.tight_layout()
fig.savefig("router_cosine_heatmaps.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved router_cosine_heatmaps.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: cross-model alignment
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, (na, nb) in zip(axes, [("baseline@27k", "deepseek@27k"),
                                 ("baseline@27k", "deepseek@39k")]):
    arr = cross_match[(na, nb)]   # (L, E)
    for L in range(N_LAYERS):
        ax.scatter([L] * N_EXPERTS, arr[L], alpha=0.3, s=8, color="#888888")
    ax.plot(layers, arr.mean(axis=1), marker="o", color="#CC4400", lw=2)
    ax.axhline(1.0, color="gray", lw=0.7, ls="--")
    ax.set_xlabel("MoE layer")
    ax.set_ylabel("max cosine sim to any expert in B")
    ax.set_title(f"Cross-model alignment\n{na}  →  {nb}")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("router_cross_alignment.png", dpi=150)
plt.close(fig)
print("Saved router_cross_alignment.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: score-bias distribution (DeepSeek only)
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, name in zip(axes, ["deepseek@27k", "deepseek@39k"]):
    sb = score_bias[name]   # (L, E)
    if sb is None:
        ax.text(0.5, 0.5, "no score_bias", ha="center", transform=ax.transAxes)
        continue
    for L in range(N_LAYERS):
        ax.scatter([L] * N_EXPERTS, sb[L], alpha=0.35, s=10)
    ax.plot(layers, sb.mean(axis=1), color="black", lw=2, label="mean")
    ax.plot(layers, sb.max(axis=1),  color="red",   lw=1, ls="--", label="max")
    ax.plot(layers, sb.min(axis=1),  color="blue",  lw=1, ls="--", label="min")
    ax.set_xlabel("MoE layer"); ax.set_ylabel("score_bias value")
    ax.set_title(f"Score-bias per expert — {name}\n(spread = better load balance)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("router_score_bias.png", dpi=150)
plt.close(fig)
print("Saved router_score_bias.png")

# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: NN cosine sim histograms (baseline vs deepseek@39k)
# ──────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(12, 10))
axes = axes.flatten()
for L in range(N_LAYERS):
    ax = axes[L]
    for name, color in [("baseline@27k", COLORS["baseline@27k"]),
                         ("deepseek@39k", COLORS["deepseek@39k"])]:
        nn = metrics[name]["nn_cosine"][L]
        ax.hist(nn, bins=20, alpha=0.6, color=color, label=name, density=True)
    ax.set_title(f"Layer {L}")
    ax.set_xlabel("NN cosine sim")
    ax.legend(fontsize=6)

fig.suptitle("Nearest-neighbour cosine similarity distribution per layer\n"
             "(left = experts are more separated)", y=1.01)
fig.tight_layout()
fig.savefig("router_nn_histograms.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("Saved router_nn_histograms.png")

# ──────────────────────────────────────────────────────────────────────────────
# Print summary table
# ──────────────────────────────────────────────────────────────────────────────

print("\n── Mean metrics (across all 9 layers) ──")
fmt = "{:<18s}  mpc={:.4f}  nn={:.4f}  eff_rank={:.2f}"
for name in CKPTS:
    m = metrics[name]
    print(fmt.format(
        name,
        m["mean_pairwise_cosine"].mean(),
        m["nn_cosine"].mean(),
        m["effective_rank"].mean(),
    ))

print("\n── Cross-model alignment ──")
for (na, nb), arr in cross_match.items():
    print(f"  {na} → {nb}: mean best-match cosine = {arr.mean():.4f}  "
          f"(per layer: {arr.mean(axis=1).round(3)})")

print("\nDone.")
