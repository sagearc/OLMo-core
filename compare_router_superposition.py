"""
Router vector geometry through a superposition lens.

Since E=64 experts live in D=1024 dims (E < D), perfect orthogonality is
geometrically achievable. The question is how far each model is from that ideal.

Reference: Elhage et al. "Toy Models of Superposition" (2022).
Interference between features i,j = cos²(wᵢ, wⱼ).
Random-vectors baseline: E[mean squared cosine] = (E-1)/D ≈ 63/1024 ≈ 0.062.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).parent / "src"))
from olmo_core.distributed.checkpoint import load_keys

N_LAYERS  = 9
N_EXPERTS = 64
D_MODEL   = 1024
RANDOM_MSC = (N_EXPERTS - 1) / D_MODEL   # ≈ 0.0615

CKPTS = {
    "baseline@27k": "runs/moe-1b-269439-baseline/step27500/model_and_optim",
    "deepseek@27k":  "runs/moe-1b-269440-deepseek/step27500/model_and_optim",
    "deepseek@39k":  "runs/moe-1b-269440-deepseek/step39750/model_and_optim",
}
COLORS = {
    "baseline@27k": "#4477AA",
    "deepseek@27k":  "#EE6677",
    "deepseek@39k":  "#AA3377",
}

# ── Load ──────────────────────────────────────────────────────────────────────

def load_router(ckpt: str) -> np.ndarray:
    """(N_LAYERS, N_EXPERTS, D_MODEL) float32, rows L2-normalized."""
    keys = [f"model.blocks.{L}.feed_forward_moe.router.weight" for L in range(N_LAYERS)]
    raw = np.stack([w.reshape(N_EXPERTS, D_MODEL).float().numpy()
                    for w in load_keys(ckpt, keys)])            # (L, E, D)
    norms = np.linalg.norm(raw, axis=2, keepdims=True)
    return raw / (norms + 1e-8)                                 # row-normalized

print("Loading...")
router = {n: load_router(p) for n, p in CKPTS.items()}
print("  done.")

# ── Per-layer metrics ─────────────────────────────────────────────────────────

def gram_eigenvalues(Wn: np.ndarray) -> np.ndarray:
    """Gram matrix G = Wn @ Wn.T, eigenvalues sorted descending. (E,)"""
    G = Wn @ Wn.T
    ev = np.linalg.eigvalsh(G)        # ascending
    return ev[::-1].copy()            # descending, sum = E

def coherence(Wn: np.ndarray) -> float:
    """max |cos(wᵢ, wⱼ)| for i≠j."""
    S = Wn @ Wn.T
    np.fill_diagonal(S, 0.0)
    return float(np.abs(S).max())

def mean_sq_cosine(Wn: np.ndarray) -> float:
    """Mean cos²(wᵢ,wⱼ) over all i≠j pairs."""
    S = Wn @ Wn.T
    off = S**2
    np.fill_diagonal(off, 0.0)
    E = Wn.shape[0]
    return float(off.sum() / (E * (E - 1)))

def gram_eff_rank(ev: np.ndarray) -> float:
    """Effective rank of Gram matrix = exp(entropy of λ/E)."""
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))

metrics = {}
for name, W in router.items():
    coh, msc, er, ev_all = [], [], [], []
    for L in range(N_LAYERS):
        ev = gram_eigenvalues(W[L])
        coh.append(coherence(W[L]))
        msc.append(mean_sq_cosine(W[L]))
        er.append(gram_eff_rank(ev))
        ev_all.append(ev)
    metrics[name] = {
        "coherence":     np.array(coh),
        "mean_sq_cosine": np.array(msc),
        "gram_eff_rank":  np.array(er),
        "gram_eigenvalues": np.array(ev_all),   # (L, E)
    }

layers = np.arange(N_LAYERS)

# ── Figure 1: scalar metrics across layers ───────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
for name, col in COLORS.items():
    ax.plot(layers, metrics[name]["coherence"], marker="o", color=col, label=name)
ax.set_xlabel("MoE layer"); ax.set_ylabel("coherence  max|cos|")
ax.set_title("Coherence (lower = more orthogonal)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.axhline(RANDOM_MSC, color="black", ls="--", lw=1.2, label=f"random ({RANDOM_MSC:.3f})")
for name, col in COLORS.items():
    ax.plot(layers, metrics[name]["mean_sq_cosine"], marker="o", color=col, label=name)
ax.set_xlabel("MoE layer"); ax.set_ylabel("mean cos²(i,j) for i≠j")
ax.set_title("Mean-squared cosine\n(= interference; random baseline shown)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
ax.axhline(N_EXPERTS, color="gray", ls=":", lw=1, label="max (64)")
for name, col in COLORS.items():
    ax.plot(layers, metrics[name]["gram_eff_rank"], marker="o", color=col, label=name)
ax.set_xlabel("MoE layer"); ax.set_ylabel("Gram eff. rank")
ax.set_title("Effective rank of row-normalised Gram\n(max = 64 = perfect orthogonality)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig("router_superposition_metrics.png", dpi=150)
plt.close(fig)
print("Saved router_superposition_metrics.png")

# ── Figure 2: Gram eigenvalue spectra for 3 layers ───────────────────────────

SEL = [0, 4, 8]
fig, axes = plt.subplots(1, len(SEL), figsize=(5 * len(SEL), 4))
for ax, L in zip(axes, SEL):
    for name, col in COLORS.items():
        ev = metrics[name]["gram_eigenvalues"][L]
        ax.plot(np.arange(1, N_EXPERTS + 1), ev, color=col, lw=1.5, label=name)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="perfect ortho (λ=1)")
    ax.set_xlabel("rank"); ax.set_ylabel("eigenvalue")
    ax.set_title(f"Gram eigenspectrum  layer {L}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

fig.suptitle("Row-normalised Gram eigenspectra (flat at 1.0 = perfectly orthogonal experts)",
             y=1.02)
fig.tight_layout()
fig.savefig("router_gram_spectra.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved router_gram_spectra.png")

# ── Figure 3: PCA scatter of 64 router vectors ───────────────────────────────

SHOW_LAYER = 4
SHOW_MODELS = ["baseline@27k", "deepseek@39k"]
fig, axes = plt.subplots(1, len(SHOW_MODELS), figsize=(5 * len(SHOW_MODELS), 5))

for ax, name in zip(axes, SHOW_MODELS):
    Wn = router[name][SHOW_LAYER]                   # (64, 1024)
    U, s, _ = np.linalg.svd(Wn, full_matrices=False)
    pct = (s[:2]**2 / (s**2).sum() * 100)
    xy = U[:, :2] * s[:2]                           # project onto top-2 PCs

    ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(N_EXPERTS),
               cmap="tab20", s=40, edgecolors="k", lw=0.3)
    for i in range(N_EXPERTS):
        ax.annotate(str(i), xy[i], fontsize=5, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_title(f"{name}  layer {SHOW_LAYER}\n"
                 f"PC1={pct[0]:.1f}%  PC2={pct[1]:.1f}% of variance")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)

fig.suptitle("PCA of 64 row-normalised router vectors (layer 4)\n"
             "Spread = diverse routing directions", y=1.02)
fig.tight_layout()
fig.savefig("router_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved router_pca_scatter.png")

# ── Figure 4: interference ratio vs random, all layers ───────────────────────

fig, ax = plt.subplots(figsize=(7, 4))
for name, col in COLORS.items():
    ratio = metrics[name]["mean_sq_cosine"] / RANDOM_MSC
    ax.plot(layers, ratio, marker="o", color=col, label=name)
ax.axhline(1.0, color="black", ls="--", lw=1, label="random-vectors baseline")
ax.set_xlabel("MoE layer")
ax.set_ylabel("mean cos²  /  random baseline")
ax.set_title("Interference ratio relative to random unit vectors in D=1024\n"
             "(1.0 = no worse than random; lower = more orthogonal)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("router_interference_ratio.png", dpi=150)
plt.close(fig)
print("Saved router_interference_ratio.png")

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\nRandom-vectors baseline: mean_sq_cosine = {RANDOM_MSC:.4f}  (= (E-1)/D = 63/1024)")
print(f"\n{'Model':<18s}  {'coherence':>10s}  {'mean_cos²':>10s}  "
      f"{'×random':>8s}  {'gram_eff_rank':>13s}")
print("-" * 70)
for name in CKPTS:
    m = metrics[name]
    mc  = m["coherence"].mean()
    msc = m["mean_sq_cosine"].mean()
    er  = m["gram_eff_rank"].mean()
    print(f"{name:<18s}  {mc:10.4f}  {msc:10.4f}  {msc/RANDOM_MSC:8.1f}×  {er:13.2f}")
