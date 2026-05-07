#!/usr/bin/env python3
"""
Three questions about the per-subcategory MoE routing produced by
experiment_wiki_routing.py:

  Q1. Are there "stars" — experts that the SAME category's subcategories
      tend to route to disproportionately often?
  Q2. Do subcategories within a category route to similar top-K sets?
  Q3. Of a subcategory's top-K, how many are "always-on" experts that fire
      for almost every subcategory vs experts that are selective to a few?

Q1/Q2 use a SHUFFLED-LABEL null: keep all subcat routings, but randomly
permute the (subcategory -> category) mapping. The null preserves marginals
(category sizes, the global routing distribution) and only kills the
within-category structure — so any gap between observed and shuffled is
real category-level signal.

Q3 is a global property — no null needed. We plot the rank-frequency curve
of expert popularity per layer, and the per-subcategory count of
"high-popularity experts in its top-K".
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).parent
NPZ = REPO / "runs/moe-1b-269440-deepseek/wiki_routing_subcats_step39750.npz"
OUT_FIG = REPO / "category_routing.png"
HEATMAP_FIG = REPO / "category_star_heatmap.png"
POPULARITY_FIG = REPO / "expert_popularity.png"

MIN_SUBCATS = 10            # categories with >= this many unique subcats
TOP_N_HEATMAP = 30          # how many largest categories to show in the heatmap
RNG_SEED = 0
ALWAYSON_THRESH = 0.50      # expert is "always-on" if it's in top-K of >= this fraction of subcats


# ── data ─────────────────────────────────────────────────────────────────────


def load_df():
    """Reproduce title.ipynb pipeline + join routing."""
    df = pl.read_csv(REPO / "children_cats.csv", has_header=False,
                     new_columns=["category", "subcategory"])
    df = df.filter(~pl.col("category").str.contains(r"\d"))
    df = df.filter(~pl.col("subcategory").str.contains(r"\d"))
    df = df.filter(pl.col("category").str.contains(r"^[a-zA-Z_]+$"))
    drop_kw = ["wiki", "importance", "stub", "article", "page", "redirect",
               "disambiguation", "all", "category", "class", "need", "template",
               "use", "navigation", "maintenance", "orphan", "uncategorized",
               "missing", "good", "feature", "articles", "with"]
    df = df.filter(~df["category"].str.contains_any(drop_kw, ascii_case_insensitive=True))
    df = df.filter(~df["subcategory"].str.contains_any(drop_kw, ascii_case_insensitive=True))
    df = (df.with_columns(pl.col("subcategory").str.split(" "))
            .explode("subcategory").drop_nulls())
    df = df.with_columns(pl.col("category", "subcategory").str.replace_all("_", " "))
    df = (df.with_columns(
            cat_words=pl.col("category").str.to_lowercase().str.split(" "),
            page_words=pl.col("subcategory").str.to_lowercase().str.split(" "),
          ).with_columns(
            score=pl.col("cat_words").list.set_intersection(pl.col("page_words")).list.len()
                  / pl.col("cat_words").list.len()
          ).filter(pl.col("score") < 0.25).drop("cat_words", "page_words", "score"))

    d = np.load(NPZ, allow_pickle=True)
    titles = list(d["titles"])
    title_to_row = {t: i for i, t in enumerate(titles)}
    expert_indices = d["expert_indices"]            # (N_sub, L, K) uint8
    n_layers = expert_indices.shape[1]
    n_experts = 64
    return df, expert_indices, title_to_row, n_layers, n_experts


def cat_to_subrows(df, title_to_row):
    """Returns dict: category -> sorted list of unique row indices into expert_indices."""
    out = {}
    for cat, sub in df.select("category", "subcategory").unique().iter_rows():
        r = title_to_row.get(sub)
        if r is None:
            continue
        out.setdefault(cat, []).append(r)
    return {c: sorted(set(rs)) for c, rs in out.items()}


# ── metrics per category, per layer ──────────────────────────────────────────


def max_expert_share(rows: np.ndarray, expert_indices: np.ndarray, layer: int, n_experts: int) -> float:
    """Fraction of subcats whose top-K at `layer` contains the single most-shared expert."""
    ei = expert_indices[rows, layer, :]                   # (n_sub, K)
    counts = np.bincount(ei.ravel(), minlength=n_experts)
    return counts.max() / len(rows)


def mean_pairwise_jaccard(rows: np.ndarray, expert_indices: np.ndarray, layer: int, max_pairs=20000, rng=None) -> float:
    """Mean Jaccard of the top-K sets across subcat pairs in this category.
    Subsamples pairs above max_pairs to keep cost bounded."""
    ei = expert_indices[rows, layer, :]                   # (n, K)
    n = ei.shape[0]
    K = ei.shape[1]
    n_pairs = n * (n - 1) // 2
    if n_pairs == 0:
        return float("nan")
    if n_pairs <= max_pairs:
        idx_pairs = list(combinations(range(n), 2))
    else:
        rng = rng or np.random.default_rng(0)
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        ok = i != j
        idx_pairs = list(zip(i[ok], j[ok]))
    sets = [frozenset(int(x) for x in row) for row in ei]
    total = 0.0
    cnt = 0
    for a, b in idx_pairs:
        sa, sb = sets[a], sets[b]
        u = len(sa | sb)
        if u == 0:
            continue
        total += len(sa & sb) / u
        cnt += 1
    return total / max(cnt, 1)


def compute_metrics(cat_rows: dict, expert_indices: np.ndarray, n_layers: int, n_experts: int):
    """Returns: cats (sorted by size desc), n_subcats[cats], stars[cats, L], jaccard[cats, L]."""
    cats = sorted(cat_rows.keys(), key=lambda c: -len(cat_rows[c]))
    cats = [c for c in cats if len(cat_rows[c]) >= MIN_SUBCATS]
    n_sub = np.array([len(cat_rows[c]) for c in cats])
    stars = np.empty((len(cats), n_layers))
    jacc = np.empty((len(cats), n_layers))
    rng = np.random.default_rng(RNG_SEED)
    for i, c in enumerate(cats):
        rows = np.asarray(cat_rows[c])
        for L in range(n_layers):
            stars[i, L] = max_expert_share(rows, expert_indices, L, n_experts)
            jacc[i, L] = mean_pairwise_jaccard(rows, expert_indices, L, rng=rng)
    return cats, n_sub, stars, jacc


def shuffled_null(cat_rows: dict, expert_indices: np.ndarray, n_layers: int, n_experts: int, seed: int):
    """Permute the subcat -> category assignment (preserving sizes) and recompute."""
    rng = np.random.default_rng(seed)
    cats = list(cat_rows.keys())
    sizes = [len(cat_rows[c]) for c in cats]
    all_rows = np.concatenate([np.asarray(cat_rows[c]) for c in cats])
    rng.shuffle(all_rows)
    out = {}
    pos = 0
    for c, s in zip(cats, sizes):
        out[c] = all_rows[pos:pos + s].tolist()
        pos += s
    return compute_metrics(out, expert_indices, n_layers, n_experts)


# ── plots ────────────────────────────────────────────────────────────────────


def plot_summary(stars, jacc, stars_null, jacc_null, n_layers: int, out_path: Path, n_cats: int):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    layers = np.arange(n_layers)

    def boxes(ax, real, null, title, ylabel):
        bp_real = ax.boxplot([real[:, L] for L in layers], positions=layers - 0.18, widths=0.32,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor="#1f77b4", alpha=0.6),
                             medianprops=dict(color="black"))
        bp_null = ax.boxplot([null[:, L] for L in layers], positions=layers + 0.18, widths=0.32,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor="#aaaaaa", alpha=0.5),
                             medianprops=dict(color="black"))
        ax.set_xticks(layers)
        ax.set_xticklabels([str(L) for L in layers])
        ax.set_xlabel("MoE layer")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend([bp_real["boxes"][0], bp_null["boxes"][0]],
                  [f"observed (n={n_cats} categories)", "shuffled-label null"],
                  loc="upper right", fontsize=9)
        ax.grid(alpha=0.3)

    boxes(ax1, stars, stars_null,
          "Q1. 'Star' expert dominance per category",
          "max expert share among category's top-K choices")
    boxes(ax2, jacc, jacc_null,
          "Q2. Routing consistency within category",
          "mean pairwise Jaccard of top-K sets")

    fig.suptitle(f"Per-category MoE routing structure  ({n_cats} categories with ≥{MIN_SUBCATS} subcats)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def expert_popularity(expert_indices: np.ndarray, n_experts: int) -> np.ndarray:
    """Returns (L, E) — fraction of subcats whose top-K at layer L includes expert e."""
    N, L, K = expert_indices.shape
    out = np.zeros((L, n_experts))
    for li in range(L):
        # presence per subcat: did expert e appear in this subcat's top-K?
        ei = expert_indices[:, li, :]
        # Each subcat contributes K experts (assumed distinct within top-K — true for top-k selection).
        # bincount over the flattened (N*K) gives, for each expert, in how many top-K's it appeared.
        counts = np.bincount(ei.ravel(), minlength=n_experts)
        out[li] = counts / N
    return out


def plot_popularity(expert_indices: np.ndarray, n_experts: int, top_k: int,
                    pop: np.ndarray, out_path: Path):
    """Two panels:
    Left  — per-layer rank-frequency curve of expert popularity (sorted desc).
    Right — per-layer histogram of (#'always-on' experts in each subcat's top-K).
    """
    N, L, K = expert_indices.shape

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    cmap = plt.get_cmap("viridis", L)

    for li in range(L):
        sorted_pop = np.sort(pop[li])[::-1]
        ax1.plot(np.arange(n_experts) + 1, sorted_pop, marker="o", ms=3,
                 color=cmap(li), label=f"layer {li}")
    ax1.axhline(top_k / n_experts, color="red", linestyle="--", linewidth=1,
                label=f"uniform K/E = {top_k/n_experts:.3f}")
    ax1.axhline(ALWAYSON_THRESH, color="black", linestyle=":", linewidth=1,
                label=f"'always-on' threshold ({ALWAYSON_THRESH:.0%})")
    ax1.set_xlabel("expert rank (sorted by popularity, descending)")
    ax1.set_ylabel("fraction of subcats with this expert in top-K")
    ax1.set_title("Q3a. Expert popularity per layer")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8, ncol=2, loc="lower left")
    ax1.grid(alpha=0.3, which="both")

    # For each subcat, count #experts in its top-K with popularity >= ALWAYSON_THRESH.
    # Plot the per-layer histogram (0..K).
    bins = np.arange(K + 2) - 0.5
    width = 0.8 / L
    for li in range(L):
        always_on_set = set(np.where(pop[li] >= ALWAYSON_THRESH)[0].tolist())
        if not always_on_set:
            counts_per_subcat = np.zeros(N, dtype=np.int32)
        else:
            ei = expert_indices[:, li, :]
            mask = np.isin(ei, list(always_on_set))
            counts_per_subcat = mask.sum(axis=1)
        h, _ = np.histogram(counts_per_subcat, bins=bins)
        x = np.arange(K + 1) + (li - (L - 1) / 2) * width
        ax2.bar(x, h / N, width=width, color=cmap(li), label=f"layer {li}")
    ax2.set_xticks(np.arange(K + 1))
    ax2.set_xlabel("# 'always-on' experts in this subcat's top-K  (always-on ≡ pop ≥ "
                   f"{ALWAYSON_THRESH:.0%})")
    ax2.set_ylabel("fraction of subcats")
    ax2.set_title("Q3b. How much of top-K is 'always-on' vs selective?")
    ax2.legend(fontsize=8, ncol=2, loc="upper right")
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle("Expert popularity: 'always-on' vs selective", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_heatmap(cats, n_sub, cat_rows, expert_indices, n_experts: int, layer: int, out_path: Path):
    """For the top-N largest categories, expert frequency among their subcats at a chosen layer."""
    cats_show = cats[:TOP_N_HEATMAP]
    H = np.zeros((len(cats_show), n_experts))
    for i, c in enumerate(cats_show):
        rows = np.asarray(cat_rows[c])
        ei = expert_indices[rows, layer, :]
        counts = np.bincount(ei.ravel(), minlength=n_experts)
        H[i] = counts / len(rows)
    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(H, aspect="auto", cmap="viridis", vmin=0, vmax=H.max())
    ax.set_xlabel("expert id")
    ax.set_yticks(np.arange(len(cats_show)))
    ax.set_yticklabels([f"{c[:50]:50s} (n={n_sub[i]})" for i, c in enumerate(cats_show)],
                       fontsize=8, family="monospace")
    ax.set_xticks(np.arange(0, n_experts, 4))
    ax.set_title(f"Per-category expert frequency at MoE layer {layer}\n"
                 f"(value = fraction of category's subcats that route to expert e in top-K)")
    fig.colorbar(im, ax=ax, label="fraction of subcats")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heatmap-layer", type=int, default=8, help="layer to use for the per-category heatmap")
    p.add_argument("--null-seed", type=int, default=42)
    args = p.parse_args()

    print("Loading data ...")
    df, expert_indices, title_to_row, n_layers, n_experts = load_df()
    cat_rows = cat_to_subrows(df, title_to_row)
    print(f"  {len(cat_rows)} categories total; "
          f"{sum(1 for v in cat_rows.values() if len(v) >= MIN_SUBCATS)} with ≥{MIN_SUBCATS} subcats")

    print("Computing observed metrics ...")
    cats, n_sub, stars, jacc = compute_metrics(cat_rows, expert_indices, n_layers, n_experts)
    print(f"  observed: cats={len(cats)}  layers={n_layers}")
    print(f"  stars  median per layer: {np.round(np.median(stars, axis=0), 3)}")
    print(f"  jacc   median per layer: {np.round(np.median(jacc, axis=0), 3)}")

    print("Computing shuffled-label null ...")
    _, _, stars_null, jacc_null = shuffled_null(cat_rows, expert_indices, n_layers, n_experts, args.null_seed)
    print(f"  null   stars median per layer:   {np.round(np.median(stars_null, axis=0), 3)}")
    print(f"  null   jaccard median per layer: {np.round(np.median(jacc_null, axis=0), 3)}")

    plot_summary(stars, jacc, stars_null, jacc_null, n_layers, OUT_FIG, n_cats=len(cats))
    plot_heatmap(cats, n_sub, cat_rows, expert_indices, n_experts, args.heatmap_layer, HEATMAP_FIG)

    print("Computing expert popularity (Q3) ...")
    pop = expert_popularity(expert_indices, n_experts)   # (L, E)
    top_k = expert_indices.shape[2]
    n_alwayson_per_layer = (pop >= ALWAYSON_THRESH).sum(axis=1)
    print(f"  #'always-on' experts (pop ≥ {ALWAYSON_THRESH:.0%}) per layer: {n_alwayson_per_layer.tolist()}")
    print(f"  max popularity per layer: {np.round(pop.max(axis=1), 3).tolist()}")
    plot_popularity(expert_indices, n_experts, top_k, pop, POPULARITY_FIG)


if __name__ == "__main__":
    main()
