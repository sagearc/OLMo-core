#!/usr/bin/env python3
"""
Singular subspace index alignment between expert weight subspaces and router directions
(following SD-MoE, https://arxiv.org/abs/2602.12556).

For each MoE layer L and each expert i:
  1. SVD the expert weight matrices  W in {w1 (gate_proj), w3 (up_proj), w2 (down_proj)},
     each (H, D) per expert.  Right singular vectors v_j^(i) in R^D form an orthonormal
     basis of the rank-H input/output subspace.
  2. Take the per-expert *router direction* r_i in R^D and unit-normalize (hat r_i):
       - centroid router: r_i = _centroid[i]   (EMA mean of routed tokens)
       - linear (default) router: r_i = weight[i]   (i-th row of the gate matrix —
         this is the "w_i" of SD-MoE Eq. 2)
  3. Index alignment: a_{i,j} = <hat r_i, v_j^(i)>^2  for j = 1..H.
     Mass concentrated at small j  => router direction aligns with the leading
     (top singular value) directions of the expert. Baseline: a_j = 1/D for a
     uniformly random unit vector.

Per layer, for each of {gate, up, down}:
  - a_j averaged over experts (log-y) vs j               [where the mass lives]
  - cumulative cumsum_{<=j} a_j averaged over experts    [how fast it concentrates]
  - top-1% / top-10% / rest interval bar                 [SD-MoE Eq.2 style]

Usage:
    uv run python analyze_subspace_alignment.py --run <run_name> --step <STEP>
    uv run python analyze_subspace_alignment.py --run moe-1b-284236-deepseek_v3 --step 21000
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from olmo_core.distributed.checkpoint import get_checkpoint_metadata, load_keys

RUNS_DIR = Path(__file__).parent / "runs"
PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")  # fixed order; w1, w3, w2


@dataclass(frozen=True)
class Shapes:
    num_experts: int
    hidden_size: int
    d_model: int


def read_shapes(run_dir: Path, step: int) -> tuple[Shapes, str]:
    cfg = json.loads((run_dir / f"step{step}" / "config.json").read_text())
    moe = cfg["model"]["block"]["feed_forward_moe"]
    shapes = Shapes(
        num_experts=int(moe["num_experts"]),
        hidden_size=int(moe["hidden_size"]),
        d_model=int(cfg["model"]["d_model"]),
    )
    return shapes, str(moe["router"]["name"])


# Routers we know how to extract a per-expert direction from. Maps router name ->
# (state-dict key suffix, expected raw shape given Shapes).
ROUTER_KEYS = {
    "centroid": ("router._centroid", lambda s: (s.num_experts, s.d_model)),
    "default": ("router.weight", lambda s: (s.num_experts * s.d_model,)),
}


def discover_moe_layers(ckpt_dir: str, router_name: str) -> list[int]:
    suffix, _ = ROUTER_KEYS[router_name]
    md = get_checkpoint_metadata(ckpt_dir)
    layers = set()
    needle = f".feed_forward_moe.{suffix}"
    for k in md.state_dict_metadata:
        if k.endswith(needle):
            parts = k.split(".")
            layers.add(int(parts[parts.index("blocks") + 1]))
    return sorted(layers)


def load_layer(ckpt_dir: str, layer: int, shapes: Shapes, router_name: str):
    """Returns (w1, w2, w3, router_dirs) all shaped per-expert; router_dirs is (E, D)."""
    base = f"model.blocks.{layer}.feed_forward_moe"
    suffix, expected_shape_fn = ROUTER_KEYS[router_name]
    keys = [
        f"{base}.experts.mlp.w1",
        f"{base}.experts.mlp.w2",
        f"{base}.experts.mlp.w3",
        f"{base}.{suffix}",
    ]
    w1, w2, w3, router = load_keys(ckpt_dir, keys)
    eh, d = shapes.num_experts * shapes.hidden_size, shapes.d_model
    for name, t in [("w1", w1), ("w2", w2), ("w3", w3)]:
        if t.shape != (eh, d):
            raise ValueError(f"layer {layer} {name}: expected {(eh, d)}, got {tuple(t.shape)}")
    expected = expected_shape_fn(shapes)
    if router.shape != expected:
        raise ValueError(
            f"layer {layer} {suffix}: expected {expected}, got {tuple(router.shape)}"
        )
    w1 = w1.float().reshape(shapes.num_experts, shapes.hidden_size, d)
    w2 = w2.float().reshape(shapes.num_experts, shapes.hidden_size, d)
    w3 = w3.float().reshape(shapes.num_experts, shapes.hidden_size, d)
    router_dirs = router.float().reshape(shapes.num_experts, d)
    return w1, w2, w3, router_dirs


@torch.no_grad()
def alignment_per_expert(
    W_eHD: torch.Tensor, centroid_ED: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    For each expert e, compute a_{e,j} = <hat c_e, v_j>^2, j = 1..H.
    Returns tensor of shape (E, H) on CPU.
    """
    W_eHD = W_eHD.to(device, non_blocking=True)
    centroid_ED = centroid_ED.to(device, non_blocking=True)
    c_unit = F.normalize(centroid_ED, dim=-1)
    _, _, Vh = torch.linalg.svd(W_eHD, full_matrices=False)  # Vh[e, j, :] = v_j^(e) in R^D
    proj = (Vh * c_unit.unsqueeze(1)).sum(dim=-1)
    return proj.pow(2).cpu()


COLORS = {"gate_proj": "tab:blue", "up_proj": "tab:orange", "down_proj": "tab:green"}


def plot_layer_panel(
    ax_curve, ax_cum, ax_bar, alignments: dict[str, np.ndarray], layer: int, baseline: np.ndarray
):
    H = alignments[PROJ_NAMES[0]].shape[1]
    D = round(1.0 / baseline[0])
    js = np.arange(1, H + 1)

    for name in PROJ_NAMES:
        ax_curve.plot(js, alignments[name].mean(axis=0), color=COLORS[name], label=name, linewidth=1.6)
    ax_curve.plot(js, baseline, "--", color="grey", linewidth=1.1, label=f"random ($1/D={1/D:.4f}$)")
    ax_curve.set_yscale("log")
    ax_curve.set_xlabel("singular index $j$ (1 = top σ)")
    ax_curve.set_ylabel(r"$\langle \hat c_i, v_j^{(i)} \rangle^2$ (mean over experts)")
    ax_curve.set_title(f"Layer {layer}: index alignment")
    ax_curve.legend(fontsize=8, loc="upper right")
    ax_curve.grid(alpha=0.3, which="both")

    for name in PROJ_NAMES:
        ax_cum.plot(js, alignments[name].mean(axis=0).cumsum(), color=COLORS[name], label=name, linewidth=1.6)
    ax_cum.plot(js, baseline.cumsum(), "--", color="grey", linewidth=1.1, label="random")
    ax_cum.axhline(1.0, color="black", linewidth=0.5, linestyle=":")
    ax_cum.set_xlabel("singular index $j$")
    ax_cum.set_ylabel(r"cumulative $\sum_{j' \leq j} a_{j'}$")
    ax_cum.set_title(f"Layer {layer}: cumulative")
    ax_cum.set_ylim(0, 1.05)
    ax_cum.legend(fontsize=8, loc="lower right")
    ax_cum.grid(alpha=0.3)

    edges = [0, max(1, H // 100), max(2, H // 10), H]
    labels = [f"top {edges[1]}\n(top 1%)", f"{edges[1]}-{edges[2]}\n(1–10%)", f"{edges[2]}-{H}\n(10–100%)"]
    width = 0.25
    x = np.arange(3)
    for k, name in enumerate(PROJ_NAMES):
        mean_a = alignments[name].mean(axis=0)
        bins = [mean_a[s:e].sum() for s, e in zip(edges[:-1], edges[1:])]
        ax_bar.bar(x + (k - 1) * width, bins, width=width, color=COLORS[name], label=name)
    rand_bins = [baseline[s:e].sum() for s, e in zip(edges[:-1], edges[1:])]
    for xi, rb in zip(x, rand_bins):
        ax_bar.hlines(rb, xi - 1.5 * width, xi + 1.5 * width, colors="grey", linestyles="--", linewidth=1.0)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_ylabel("energy fraction")
    ax_bar.set_title(f"Layer {layer}: interval energy")
    ax_bar.legend(fontsize=8, loc="upper right")
    ax_bar.grid(alpha=0.3, axis="y")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="moe-1b-283608-ema_centroid")
    p.add_argument("--step", type=int, default=21000)
    p.add_argument("--out", default=None)
    p.add_argument("--layers", type=int, nargs="*", default=None,
                   help="MoE layer indices to analyze; default = all")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="torch device for the batched SVD (default: cuda if available, else cpu)")
    args = p.parse_args()

    run_dir = RUNS_DIR / args.run
    if not run_dir.is_dir():
        sys.exit(f"Run not found: {run_dir}")
    ckpt_dir = run_dir / f"step{args.step}" / "model_and_optim"
    if not ckpt_dir.is_dir():
        sys.exit(f"Checkpoint not found: {ckpt_dir}")
    shapes, router_name = read_shapes(run_dir, args.step)
    if router_name not in ROUTER_KEYS:
        sys.exit(f"Unsupported router '{router_name}'; known: {sorted(ROUTER_KEYS)}")
    device = torch.device(args.device)
    print(f"Run: {args.run}  step: {args.step}  device: {device}  router: {router_name}")
    print(f"Shapes: E={shapes.num_experts} H={shapes.hidden_size} D={shapes.d_model}")

    moe_layers = discover_moe_layers(str(ckpt_dir), router_name)
    print(f"MoE layers in checkpoint: {moe_layers}")
    if args.layers is not None:
        moe_layers = [L for L in moe_layers if L in set(args.layers)]
    print(f"Analyzing layers: {moe_layers}")

    # E[<g, e_j>^2] = 1/D for a uniformly random unit vector g in R^D, every j.
    baseline = np.full(shapes.hidden_size, 1.0 / shapes.d_model)

    per_layer: dict[int, dict[str, np.ndarray]] = {}
    for L in moe_layers:
        print(f"  ── layer {L} ── loading + SVD ...")
        w1, w2, w3, router_dirs = load_layer(str(ckpt_dir), L, shapes, router_name)
        per_layer[L] = {
            "gate_proj": alignment_per_expert(w1, router_dirs, device).numpy(),
            "up_proj": alignment_per_expert(w3, router_dirs, device).numpy(),
            "down_proj": alignment_per_expert(w2, router_dirs, device).numpy(),
        }
        del w1, w2, w3, router_dirs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        for name in PROJ_NAMES:
            a = per_layer[L][name]
            top1 = a[:, : max(1, shapes.hidden_size // 100)].sum(axis=1).mean()
            tot = a.sum(axis=1).mean()
            print(f"    {name}: mean top-1% energy = {top1:.4f},  total in subspace = {tot:.4f}")

    n_layers = len(moe_layers)
    fig, axes = plt.subplots(n_layers, 3, figsize=(16, 3.4 * n_layers), squeeze=False)
    fig.suptitle(
        f"Singular subspace index alignment between router directions and expert subspaces\n"
        f"{args.run} · step {args.step}  ·  router={router_name}  ·  "
        f"E={shapes.num_experts}, H={shapes.hidden_size}, D={shapes.d_model}",
        fontsize=12,
    )
    for row, L in enumerate(moe_layers):
        plot_layer_panel(axes[row, 0], axes[row, 1], axes[row, 2], per_layer[L], L, baseline)
    plt.tight_layout(rect=[0, 0, 1, 0.985])

    out = args.out or str(run_dir / f"subspace_alignment_step{args.step}.png")
    plt.savefig(out, dpi=140)
    plt.close()
    print(f"Saved: {out}")

    print(f"\n── Summary (mean over experts; baseline ~ 1/D={1/shapes.d_model:.4f} per index) ──")
    print(f"{'layer':>6} | {'gate top1%':>11} {'top10%':>8} {'total':>7} | "
          f"{'up top1%':>9} {'top10%':>8} {'total':>7} | "
          f"{'down top1%':>11} {'top10%':>8} {'total':>7}")
    for L in moe_layers:
        stats: dict[str, tuple[float, float, float]] = {}
        for name in PROJ_NAMES:
            a = per_layer[L][name]
            top1 = a[:, : max(1, shapes.hidden_size // 100)].sum(axis=1).mean()
            top10 = a[:, : max(1, shapes.hidden_size // 10)].sum(axis=1).mean()
            stats[name] = (float(top1), float(top10), float(a.sum(axis=1).mean()))
        g, u, d = stats["gate_proj"], stats["up_proj"], stats["down_proj"]
        print(
            f"{L:>6d} | "
            f"{g[0]:>11.4f} {g[1]:>8.4f} {g[2]:>7.4f} | "
            f"{u[0]:>9.4f} {u[1]:>8.4f} {u[2]:>7.4f} | "
            f"{d[0]:>11.4f} {d[1]:>8.4f} {d[2]:>7.4f}"
        )

    np.savez(
        str(Path(out).with_suffix(".npz")),
        layers=np.array(moe_layers),
        **{f"L{L}_{name}": per_layer[L][name] for L in moe_layers for name in PROJ_NAMES},
        baseline=baseline,
    )
    print(f"Saved raw alignments: {Path(out).with_suffix('.npz')}")


if __name__ == "__main__":
    main()
