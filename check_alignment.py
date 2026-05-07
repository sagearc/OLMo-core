"""
Heatmap of router-expert subspace alignment.
Cell (i,j) = ||V_k(j) @ r_i||^2  (how much router i lies in top-k dirs of expert j).
"""
import sys
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent / "src"))
from olmo_core.distributed.checkpoint import load_keys

N_LAYERS, N_EXPERTS, D_MODEL, H_EXPERT = 9, 64, 1024, 512
SEL_LAYERS = [0, 4, 8]
CKPTS = {
    "With auxiliary loss": "runs/moe-1b-269439-baseline/step21000/model_and_optim",
    "Without auxiliary loss": "runs/moe-1b-269440-deepseek/step21000/model_and_optim",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def load_router(ckpt):
    keys = [f"model.blocks.{L}.feed_forward_moe.router.weight" for L in range(N_LAYERS)]
    R = torch.stack([w.reshape(N_EXPERTS, D_MODEL).float() for w in load_keys(ckpt, keys)]).to(device)
    return R / R.norm(dim=2, keepdim=True)

def load_w1(ckpt):
    keys = [f"model.blocks.{L}.feed_forward_moe.experts.mlp.w1" for L in range(N_LAYERS)]
    return torch.stack([w.reshape(N_EXPERTS, H_EXPERT, D_MODEL).float() for w in load_keys(ckpt, keys)]).to(device)

print("Loading...")
R = {n: load_router(p) for n, p in CKPTS.items()}
W = {n: load_w1(p)     for n, p in CKPTS.items()}
print("Done.")

def alignment_matrix(W_all, R_all, k):
    """Returns (N_LAYERS, N_EXPERTS, N_EXPERTS): scores[L,i,j] = ||V_k(j) @ r_i||^2"""
    scores = torch.zeros(N_LAYERS, N_EXPERTS, N_EXPERTS, device=device)
    for L in range(N_LAYERS):
        _, _, Vt = torch.linalg.svd(W_all[L], full_matrices=False)  # (E, min(H,D), D)
        Vk = Vt[:, :k, :]                                            # (E, k, D)
        proj = torch.einsum("jkd,id->jki", Vk, R_all[L])            # (E_exp, k, E_rtr)
        scores[L] = (proj ** 2).sum(dim=1).T                         # (E_rtr, E_exp)
    return scores.cpu()

for pct, k in [(1, int(torch.tensor(0.01 * min(H_EXPERT, D_MODEL)).ceil().item())),
               (5, int(torch.tensor(0.05 * min(H_EXPERT, D_MODEL)).ceil().item()))]:
    print(f"Computing top {pct}% (k={k})...")
    mats = {n: alignment_matrix(W[n], R[n], k) for n in CKPTS}

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    vmax = max(mats[n].max().item() for n in CKPTS)

    for row, name in enumerate(CKPTS):
        for col, L in enumerate(SEL_LAYERS):
            ax = axes[row, col]
            im = ax.imshow(mats[name][L].numpy(), vmin=0, vmax=vmax, cmap="YlOrRd", aspect="auto")
            ax.set_title(f"Layer {L}" if row == 0 else "", fontsize=9)
            ax.set_ylabel(name if col == 0 else "", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    fig.colorbar(im, ax=axes, shrink=0.6, label=r"$\|\|V_k^{(j)} r_i\|\|^2$")
    out = f"alignment_top{pct}pct.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
