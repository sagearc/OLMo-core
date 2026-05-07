#!/usr/bin/env python3
"""
For each MoE layer L, expert e, and w_gate hidden-unit index j ∈ [0, H), compute
the conditional probability that j appears in the top-10 of expert e for a
random subcategory THAT ACTUALLY ROUTED TO e at layer L:

   P_global(j | L, e) = P(j ∈ top-10_e | subcat picked expert e at layer L)

Conditioning on the chosen expert is essential: the H=512 hidden axis is
expert-specific, so neuron index j of expert 5 is a different weight row
from neuron index j of expert 12. Pooling across experts conflates
unrelated features.

Per-category enrichment is then  P_cat(j | L, e) / P_global(j | L, e).
A neuron is category-specific *for that expert* if its in-category firing
rate (over subcats in the category that picked e at L) is many-fold above
its global rate (over all subcats that picked e at L).

Outputs:
  - wgate_index_popularity.png   left: per-(L,e) rank curves;
                                 right: enrichment heatmap for one (L,e) cell
  - wgate_index_top.csv          leading neurons per (layer, expert) globally
  - wgate_index_cat_top.csv      most-enriched (category, neuron) for the
                                 chosen heatmap (L, e) cell
  - wgate_prevalence.csv         every (cat, L, e, j) with P_cat ≥ threshold
                                 — quantifies how often the structure shows up
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

REPO = Path(__file__).parent
NPZ = REPO / "runs/moe-1b-269440-deepseek/wiki_routing_subcats_step39750.npz"
OUT_FIG = REPO / "wgate_index_popularity.png"
OUT_GLOBAL_CSV = REPO / "wgate_index_top.csv"
OUT_CAT_CSV = REPO / "wgate_index_cat_top.csv"
OUT_PREVALENCE_CSV = REPO / "wgate_prevalence.csv"

TOP_LEADING = 10                 # how many leading neurons to print per (L, e)
MIN_SUBCATS_PER_CAT = 30         # category must have ≥ this many unique subcats overall
MIN_PICKS_PER_CAT_LE = 10        # (cat, L, e) cell needs ≥ this many subcats picking e to be analyzed
MIN_PCAT_FOR_ENRICH = 0.30       # neuron must fire in ≥ this fraction of cell subcats to count
TOP_N_CATS_HEATMAP = 25
TOP_N_NEURONS_HEATMAP = 60
EPS = 1e-6


def load_df_with_routing():
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
    return (df, title_to_row,
            d["expert_indices"].astype(np.int64),       # (N, L, K)
            d["w_gate_top10_idx"].astype(np.int32))     # (N, L, K, 10)


def popularity_per_expert(w_gate: np.ndarray, expert_indices: np.ndarray,
                           rows: np.ndarray, n_neurons: int, n_experts: int):
    """Conditional popularity over the given subcat rows.

    Returns
      pop      (L, E, H) float32  — P(neuron j ∈ top-10 of expert e | subcat picked e at L),
                                    estimated only over subcats in `rows` that picked e at L.
      n_picked (L, E)    int64    — count of subcats in `rows` that picked expert e at layer L.
    """
    L = expert_indices.shape[1]
    if len(rows) == 0:
        return np.zeros((L, n_experts, n_neurons), dtype=np.float32), np.zeros((L, n_experts), dtype=np.int64)
    sub_ei = expert_indices[rows]                       # (n, L, K)
    sub_wg = w_gate[rows]                               # (n, L, K, 10)
    pop = np.zeros((L, n_experts, n_neurons), dtype=np.float32)
    n_picked = np.zeros((L, n_experts), dtype=np.int64)
    for li in range(L):
        ei_l = sub_ei[:, li, :]                         # (n, K)
        wg_l = sub_wg[:, li, :, :]                      # (n, K, 10)
        for e in range(n_experts):
            # top-K is a set of distinct experts, so each row has at most one True.
            mask = (ei_l == e)                          # (n, K)
            n_picked[li, e] = int(mask.sum())
            if n_picked[li, e] == 0:
                continue
            # within a single (subcat, expert) the 10 indices are distinct (torch.topk):
            # plain bincount + divide gives the per-neuron firing fraction.
            sets10 = wg_l[mask]                         # (n_picked, 10)
            pop[li, e] = (np.bincount(sets10.ravel(), minlength=n_neurons).astype(np.float32)
                          / n_picked[li, e])
    return pop, n_picked


def cat_to_unique_rows(df: pl.DataFrame, title_to_row: dict) -> dict[str, np.ndarray]:
    out: dict[str, list[int]] = {}
    for cat, sub in df.select("category", "subcategory").unique().iter_rows():
        r = title_to_row.get(sub)
        if r is not None:
            out.setdefault(cat, []).append(r)
    return {c: np.asarray(sorted(set(rs))) for c, rs in out.items()}


def plot_global_and_enrichment(pop_global: np.ndarray, n_global: np.ndarray,
                               heatmap_cell: tuple[int, int],
                               cats_h: list[str], pop_cats: dict, n_cats: dict,
                               n_global_total: int, out_path: Path):
    L, E, H = pop_global.shape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    cmap = plt.get_cmap("viridis", L)
    rank = np.arange(H) + 1

    # Left: per-layer rank-popularity curve, median and p25–p75 over experts that
    # are actually picked enough times to give a stable estimate.
    threshold = max(1, int(0.005 * n_global_total))     # at least 0.5% of subcats picked it
    for li in range(L):
        keep = n_global[li] >= threshold
        if not keep.any():
            continue
        arr = np.sort(pop_global[li][keep], axis=1)[:, ::-1]
        med = np.median(arr, axis=0)
        p25 = np.percentile(arr, 25, axis=0)
        p75 = np.percentile(arr, 75, axis=0)
        c = cmap(li)
        ax1.fill_between(rank, p25, p75, color=c, alpha=0.15)
        ax1.plot(rank, med, color=c, label=f"layer {li}")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("neuron rank within expert (sorted by popularity, descending)")
    ax1.set_ylabel("P(neuron ∈ top-10 of expert | subcat picked this expert)")
    ax1.set_title(f"Per-(layer, expert) w_gate index popularity\n"
                  f"median ± p25–p75 across experts with ≥ {threshold} pickers")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend(fontsize=8, loc="lower left", ncol=2)

    # Right: enrichment heatmap for ONE specific (L, e) cell.
    li, e = heatmap_cell
    p_g = pop_global[li, e] + EPS
    cat_top_idx: set[int] = set()
    for c in cats_h:
        cat_top_idx.update(np.argsort(pop_cats[c][li, e])[::-1][:TOP_N_NEURONS_HEATMAP].tolist())
    cat_top_idx.update(np.argsort(p_g)[::-1][:TOP_N_NEURONS_HEATMAP].tolist())
    cols = sorted(cat_top_idx)

    enr = np.zeros((len(cats_h), len(cols)), dtype=np.float32)
    for i, c in enumerate(cats_h):
        enr[i] = (pop_cats[c][li, e, cols] + EPS) / p_g[cols]
    log2enr = np.log2(np.clip(enr, 1e-3, 1e3))

    im = ax2.imshow(log2enr, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax2.set_xticks(np.arange(0, len(cols), max(1, len(cols)//40)))
    ax2.set_xticklabels(np.array(cols)[::max(1, len(cols)//40)], rotation=90, fontsize=7)
    ax2.set_xlabel(f"w_gate neuron index (showing {len(cols)} popular ones)")
    ax2.set_yticks(np.arange(len(cats_h)))
    ax2.set_yticklabels([f"{c[:46]} ({n_cats[c][li,e]})" for c in cats_h],
                       fontsize=8, family="monospace")
    ax2.set_title(f"Enrichment at (layer {li}, expert {e}) — n_global={n_global[li,e]}\n"
                  f"log₂(P_cat / P_global) — both conditioned on picking this expert")
    fig.colorbar(im, ax=ax2, label="log₂ enrichment")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def report_prevalence(prev: pl.DataFrame, n_cells_analyzable: int, n_cats_total: int,
                      n_layers: int) -> None:
    """Quantify how often per-(category, layer, expert) cells show selective neuron firing."""
    if prev.is_empty():
        print("\nPrevalence: no (cat, L, e, j) tuples passed the filter."); return

    print(f"\n── Prevalence of category-specific within-expert structure ──")
    print(f"  scope: {n_cells_analyzable} (cat, L, e) cells with "
          f"≥ {MIN_PICKS_PER_CAT_LE} subcats picking that expert at that layer")
    print(f"  filter: P_cat ≥ {MIN_PCAT_FOR_ENRICH} (≈ {MIN_PCAT_FOR_ENRICH*MIN_PICKS_PER_CAT_LE:.0f}+ "
          f"of the cell's subcats fire that neuron)")
    print(f"  total (cat, L, e, j) tuples passing filter: {prev.height}")

    cells_w_any = prev.select("category", "layer", "expert").unique().height
    print(f"  (cat, L, e) cells with ≥1 such neuron: "
          f"{cells_w_any} / {n_cells_analyzable} ({100*cells_w_any/n_cells_analyzable:.1f}%)")

    for thresh in (2, 5, 10, 20):
        sub = prev.filter(pl.col("ratio") >= thresh)
        cells = sub.select("category", "layer", "expert").unique().height
        cats = sub.select("category").unique().height
        print(f"  ratio ≥ {thresh:2d}× : {sub.height:>6d} tuples  |  "
              f"{cells:>5d}/{n_cells_analyzable} cells "
              f"({100*cells/n_cells_analyzable:5.1f}%)  |  "
              f"{cats:>4d}/{n_cats_total} unique categories "
              f"({100*cats/n_cats_total:5.1f}%)")

    strong = prev.filter((pl.col("ratio") >= 10) & (pl.col("p_cat") >= 0.5))
    s_cells = strong.select("category", "layer", "expert").unique().height
    s_cats = strong.select("category").unique().height
    print(f"\n  STRONG (ratio ≥ 10× AND P_cat ≥ 0.5):  {strong.height} tuples  "
          f"in {s_cells} cells  covering {s_cats} unique categories "
          f"({100*s_cats/n_cats_total:.1f}%)")

    print(f"  per-layer #strong tuples:")
    by_layer = (strong.group_by("layer").agg(pl.len().alias("n")).sort("layer"))
    for r in by_layer.iter_rows(named=True):
        print(f"    layer {r['layer']}: {r['n']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--heatmap-layer", type=int, default=8,
                   help="layer for the per-(L,e) heatmap (default: last)")
    p.add_argument("--heatmap-expert", type=int, default=None,
                   help="expert for the heatmap (default: most-picked at the chosen layer)")
    args = p.parse_args()

    print("Loading data ...")
    df, title_to_row, expert_indices, w_gate = load_df_with_routing()
    N_total, L, K = expert_indices.shape
    n_neurons = int(w_gate.max()) + 1
    n_experts = int(expert_indices.max()) + 1
    print(f"  N={N_total}  L={L}  K={K}  E={n_experts}  H={n_neurons}")

    print("Computing global per-expert popularity ...")
    pop_global, n_global = popularity_per_expert(
        w_gate, expert_indices, np.arange(N_total), n_neurons, n_experts
    )
    print(f"  pop_global shape {pop_global.shape}")
    print(f"  n_global per layer (min, median, max): "
          f"{n_global.min(axis=1).tolist()} / "
          f"{[int(np.median(n_global[li])) for li in range(L)]} / "
          f"{n_global.max(axis=1).tolist()}")

    print("\nLeading w_gate neurons per (layer, expert) — top 5 each, picking the busiest expert per layer:")
    rows_global = []
    for li in range(L):
        e_busy = int(np.argmax(n_global[li]))
        for rk, j in enumerate(np.argsort(pop_global[li, e_busy])[::-1][:TOP_LEADING]):
            rows_global.append({
                "layer": li, "expert": e_busy, "rank": rk + 1,
                "neuron_idx": int(j),
                "p_global": float(pop_global[li, e_busy, j]),
                "n_picked_global": int(n_global[li, e_busy]),
            })
        head = ", ".join(
            f"j={int(j)} P={pop_global[li, e_busy, j]:.3f}"
            for j in np.argsort(pop_global[li, e_busy])[::-1][:5]
        )
        print(f"  layer {li}, expert {e_busy} (n={n_global[li, e_busy]}): {head}")
    pl.DataFrame(rows_global).write_csv(OUT_GLOBAL_CSV)
    print(f"Saved: {OUT_GLOBAL_CSV}")

    print("\nComputing per-category per-expert popularity ...")
    cat_rows = cat_to_unique_rows(df, title_to_row)
    cat_rows = {c: rs for c, rs in cat_rows.items() if len(rs) >= MIN_SUBCATS_PER_CAT}
    print(f"  {len(cat_rows)} categories with ≥{MIN_SUBCATS_PER_CAT} unique subcats")

    pop_cats: dict[str, np.ndarray] = {}
    n_cats: dict[str, np.ndarray] = {}
    for c, rs in cat_rows.items():
        pop_cats[c], n_cats[c] = popularity_per_expert(w_gate, expert_indices, rs, n_neurons, n_experts)

    # Pick the heatmap expert: the one with most pickers at the chosen layer.
    li_h = args.heatmap_layer
    e_h = args.heatmap_expert if args.heatmap_expert is not None else int(np.argmax(n_global[li_h]))
    print(f"\nHeatmap cell: layer={li_h}, expert={e_h}, n_global={n_global[li_h, e_h]}")

    cats_h = [c for c in cat_rows
              if n_cats[c][li_h, e_h] >= MIN_PICKS_PER_CAT_LE]
    cats_h = sorted(cats_h, key=lambda c: -n_cats[c][li_h, e_h])[:TOP_N_CATS_HEATMAP]
    print(f"  {len(cats_h)} categories qualify for heatmap "
          f"(≥{MIN_PICKS_PER_CAT_LE} subcats picking expert {e_h} at layer {li_h})")

    # Top enriched (cat, neuron) at heatmap cell.
    p_g_cell = pop_global[li_h, e_h] + EPS
    enriched_rows = []
    for c, pop in pop_cats.items():
        if n_cats[c][li_h, e_h] < MIN_PICKS_PER_CAT_LE: continue
        ratios = (pop[li_h, e_h] + EPS) / p_g_cell
        mask = pop[li_h, e_h] >= MIN_PCAT_FOR_ENRICH
        for j in np.where(mask)[0]:
            enriched_rows.append({
                "category": c, "n_picked_in_cat": int(n_cats[c][li_h, e_h]),
                "layer": li_h, "expert": e_h, "neuron_idx": int(j),
                "p_cat": float(pop[li_h, e_h, j]),
                "p_global": float(pop_global[li_h, e_h, j]),
                "ratio": float(ratios[j]),
            })
    enriched = pl.DataFrame(enriched_rows).sort("ratio", descending=True)
    enriched.head(50).write_csv(OUT_CAT_CSV)
    print(f"Saved: {OUT_CAT_CSV}")
    print(f"\nMost enriched (cat, neuron) at layer {li_h}, expert {e_h}:")
    for r in enriched.head(15).iter_rows(named=True):
        print(f"  {r['category'][:40]:40s} neuron {r['neuron_idx']:3d}  "
              f"P_cat={r['p_cat']:.2f}  P_global={r['p_global']:.3f}  "
              f"ratio={r['ratio']:5.1f}×  (n={r['n_picked_in_cat']})")

    # ── full prevalence across all (cat, L, e) cells ──
    print("\nComputing prevalence across all (cat, L, e) cells ...")
    prev_rows = []
    n_analyzable_cells = 0
    for c, pop in pop_cats.items():
        for li in range(L):
            for e in range(n_experts):
                if n_cats[c][li, e] < MIN_PICKS_PER_CAT_LE: continue
                n_analyzable_cells += 1
                p_cell = pop[li, e]
                p_g = pop_global[li, e] + EPS
                mask = p_cell >= MIN_PCAT_FOR_ENRICH
                if not mask.any(): continue
                ratios = (p_cell + EPS) / p_g
                for j in np.where(mask)[0]:
                    prev_rows.append({
                        "category": c, "n_picked_in_cat": int(n_cats[c][li, e]),
                        "layer": int(li), "expert": int(e), "neuron_idx": int(j),
                        "p_cat": float(p_cell[j]),
                        "p_global": float(pop_global[li, e, j]),
                        "ratio": float(ratios[j]),
                    })
    prev = pl.DataFrame(prev_rows)
    prev.write_csv(OUT_PREVALENCE_CSV)
    print(f"Saved full prevalence: {OUT_PREVALENCE_CSV}  ({prev.height} tuples)")
    report_prevalence(prev, n_analyzable_cells, n_cats_total=len(cat_rows), n_layers=L)

    plot_global_and_enrichment(pop_global, n_global, (li_h, e_h),
                               cats_h, pop_cats, n_cats, N_total, OUT_FIG)


if __name__ == "__main__":
    main()
