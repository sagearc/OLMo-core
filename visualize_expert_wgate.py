"""
Visualize w1 (gate projection) rows of a single MoE expert via K-Means.

For a chosen run / step / layer / expert:
  - Load w1: shape [num_experts, hidden_size, d_model]
  - Slice the target expert: [hidden_size, d_model]  (e.g. 512 rows of 1024-dim)
  - Run K-Means with k=K
  - Project all rows to 2D with PCA
  - Scatter-plot each row colored by its cluster; mark centroids with a star

An elbow subplot (inertia vs k) is shown alongside to help pick k.

Usage:
    uv run python visualize_expert_wgate.py <run_name> [options]
    uv run python visualize_expert_wgate.py moe-1b-269032-baseline --step 250 --layer 4 --expert 0 --k 8
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# ── defaults ───────────────────────────────────────────────────────────────────
RUNS_DIR   = os.path.join(os.path.dirname(__file__), "runs")
NUM_EXPERTS = 64
DEFAULT_LAYER  = 4
DEFAULT_EXPERT = 0
DEFAULT_K      = 8
ELBOW_MAX_K    = 20


# ── checkpoint ─────────────────────────────────────────────────────────────────

def latest_step(run_dir: str) -> int:
    steps = [
        int(n[4:])
        for n in os.listdir(run_dir)
        if n.startswith("step") and os.path.isdir(os.path.join(run_dir, n))
        and n[4:].isdigit()
    ]
    if not steps:
        raise FileNotFoundError(f"No step checkpoints in {run_dir}")
    return max(steps)


def load_expert_w1(run_dir: str, step: int, layer: int, expert: int) -> np.ndarray:
    """
    Returns float32 numpy array of shape [hidden_size, d_model] for one expert.
    Uses olmo-core's load_keys (no distributed context needed).
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    from olmo_core.distributed.checkpoint import load_keys

    ckpt_dir = os.path.join(run_dir, f"step{step}", "model_and_optim")
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

    key = f"model.blocks.{layer}.feed_forward_moe.experts.mlp.w1"
    (w1_flat,) = load_keys(ckpt_dir, [key])          # [E*H, D]

    d_model = w1_flat.shape[1]
    hidden  = w1_flat.shape[0] // NUM_EXPERTS
    w1 = w1_flat.float().reshape(NUM_EXPERTS, hidden, d_model)
    return w1[expert].numpy()                         # [H, D]


# ── elbow ──────────────────────────────────────────────────────────────────────

def compute_elbow(rows: np.ndarray, max_k: int) -> list[float]:
    inertias = []
    for k in range(1, max_k + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=0)
        km.fit(rows)
        inertias.append(km.inertia_)
    return inertias


# ── plot ───────────────────────────────────────────────────────────────────────

def make_plot(
    rows: np.ndarray,       # [H, D]
    k: int,
    inertias: list[float],
    run_name: str,
    step: int,
    layer: int,
    expert: int,
    out_path: str,
) -> None:
    # K-Means in original high-dim space
    km = KMeans(n_clusters=k, n_init=10, random_state=0)
    labels = km.fit_predict(rows)
    centroids = km.cluster_centers_               # [k, D]
    cluster_sizes = np.bincount(labels, minlength=k)

    # Sort rows by cluster for the heatmap
    sort_idx = np.argsort(labels)
    sorted_rows = rows[sort_idx]                  # [H, D]

    # Cosine similarity between centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cent_normed = centroids / (norms + 1e-8)
    cosine_sim = cent_normed @ cent_normed.T      # [k, k]

    cmap_clusters = plt.get_cmap("tab20", k)
    colors = [cmap_clusters(i) for i in range(k)]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    title = f"{run_name}  ·  layer {layer}  ·  expert {expert}  ·  step {step}  ·  k={k}"
    fig.suptitle(title, fontsize=11)

    # ── left: sorted row heatmap ───────────────────────────────────────────────
    ax = axes[0]
    vmax = np.percentile(np.abs(sorted_rows), 98)
    im = ax.imshow(sorted_rows, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # draw cluster boundary lines and labels on the left
    boundary = 0
    for c in range(k):
        size = cluster_sizes[np.argsort(labels)[boundary:boundary + cluster_sizes[c]].shape[0]
                             if False else c]  # just cluster_sizes[c]
        ax.axhline(boundary - 0.5, color=colors[c], linewidth=1.2)
        ax.text(-5, boundary + size / 2, str(c), color=colors[c],
                fontsize=7, fontweight="bold", ha="right", va="center")
        boundary += size

    ax.set_title("w1 rows sorted by cluster\n(each row = one hidden unit, cols = d_model)")
    ax.set_xlabel("d_model dimension")
    ax.set_ylabel("hidden unit (sorted by cluster)")
    ax.set_yticks([])

    # ── middle: centroid cosine similarity ─────────────────────────────────────
    ax2 = axes[1]
    im2 = ax2.imshow(cosine_sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im2, ax=ax2, fraction=0.04, pad=0.02)
    for i in range(k):
        for j in range(k):
            ax2.text(j, i, f"{cosine_sim[i, j]:.2f}", ha="center", va="center",
                     fontsize=6, color="black")
    ax2.set_xticks(range(k))
    ax2.set_yticks(range(k))
    ax2.set_title("Centroid cosine similarity\n(off-diagonal → cluster distinctness)")
    ax2.set_xlabel("cluster")
    ax2.set_ylabel("cluster")

    # ── right: elbow + cluster sizes ──────────────────────────────────────────
    ax3 = axes[2]
    ks = list(range(1, len(inertias) + 1))
    ax3.plot(ks, inertias, "o-", color="steelblue", label="inertia")
    ax3.axvline(k, color="red", linestyle="--", linewidth=1, label=f"chosen k={k}")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Inertia", color="steelblue")
    ax3.set_title("Elbow  +  cluster sizes")
    ax3.legend(loc="upper right")

    ax3b = ax3.twinx()
    ax3b.bar(range(k), cluster_sizes, color=[colors[c] for c in range(k)],
             alpha=0.4, width=0.4)
    ax3b.set_ylabel("rows per cluster", color="gray")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── main ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_name")
    p.add_argument("--step",   type=int, default=None,           help="checkpoint step (default: latest)")
    p.add_argument("--layer",  type=int, default=DEFAULT_LAYER,  help=f"layer index (default: {DEFAULT_LAYER})")
    p.add_argument("--expert", type=int, default=DEFAULT_EXPERT, help=f"expert index 0-63 (default: {DEFAULT_EXPERT})")
    p.add_argument("--k",      type=int, default=DEFAULT_K,      help=f"number of K-Means clusters (default: {DEFAULT_K})")
    p.add_argument("--elbow-max-k", type=int, default=ELBOW_MAX_K)
    p.add_argument("--projection", choices=["pca", "umap"], default="pca")
    p.add_argument("--out",    default=None)
    return p.parse_args()


def main():
    args = parse_args()

    run_dir = os.path.join(RUNS_DIR, args.run_name)
    if not os.path.isdir(run_dir):
        sys.exit(f"Run not found: {run_dir}")

    step = args.step if args.step is not None else latest_step(run_dir)
    print(f"run={args.run_name}  step={step}  layer={args.layer}  expert={args.expert}  k={args.k}")

    print("Loading w1 ...")
    rows = load_expert_w1(run_dir, step, args.layer, args.expert)
    print(f"  rows shape: {rows.shape}")

    print(f"Elbow analysis (k=1..{args.elbow_max_k}) ...")
    inertias = compute_elbow(rows, args.elbow_max_k)

    out = args.out or os.path.join(
        run_dir,
        f"expert{args.expert}_layer{args.layer}_step{step}_k{args.k}.png",
    )
    make_plot(rows, k=args.k, inertias=inertias,
              run_name=args.run_name, step=step,
              layer=args.layer, expert=args.expert, out_path=out)


if __name__ == "__main__":
    main()
