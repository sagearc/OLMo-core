import logging
from typing import TYPE_CHECKING, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import distribute_like, get_local_tensor

if TYPE_CHECKING:
    from ..attention import SequenceMixer
    from ..feed_forward import FeedForward
    from ..moe import MoEBase


log = logging.getLogger(__name__)


def _apply_init(init_fun, x: torch.Tensor, *args, **kwargs):
    if not isinstance(x, DTensor):
        init_fun(x, *args, **kwargs)
        return

    # Initialize full version of x locally, then apply init to that.
    full_x = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    init_fun(full_x, *args, **kwargs)
    full_x = distribute_like(x, full_x)

    # Now copy over the corresponding shard of `full_x` into `x`.
    get_local_tensor(x).copy_(get_local_tensor(full_x))


def _balanced_fixed_centroid_assignment(scores: torch.Tensor, rows_per_expert: int) -> torch.Tensor:
    """
    Assign every row to a fixed centroid while enforcing equal expert widths.

    Rows first propose their highest-scoring centroid that still has capacity.
    An oversubscribed centroid accepts its strongest proposals; rejected rows
    propose again among the remaining centroids.
    """
    num_rows, num_experts = scores.shape
    if num_rows != num_experts * rows_per_expert:
        raise ValueError(f"expected {num_experts * rows_per_expert} rows, got {num_rows}")

    assignment = torch.full((num_rows,), -1, dtype=torch.long, device=scores.device)
    capacity = torch.full((num_experts,), rows_per_expert, dtype=torch.long, device=scores.device)
    remaining = torch.arange(num_rows, device=scores.device)

    while remaining.numel() > 0:
        available = capacity > 0
        proposal_scores = scores.index_select(0, remaining).masked_fill(
            ~available.unsqueeze(0), -torch.inf
        )
        proposals = proposal_scores.argmax(dim=1)
        accepted_parts = []

        for expert_idx in available.nonzero(as_tuple=False).flatten().tolist():
            candidate_positions = (proposals == expert_idx).nonzero(as_tuple=False).flatten()
            if candidate_positions.numel() == 0:
                continue
            candidate_rows = remaining.index_select(0, candidate_positions)
            take = min(candidate_rows.numel(), int(capacity[expert_idx].item()))
            if candidate_rows.numel() > take:
                candidate_scores = scores[candidate_rows, expert_idx]
                keep = candidate_scores.topk(take, sorted=False).indices
                candidate_rows = candidate_rows.index_select(0, keep)
            assignment[candidate_rows] = expert_idx
            capacity[expert_idx] -= take
            accepted_parts.append(candidate_rows)

        if not accepted_parts:
            raise RuntimeError("balanced router-centroid assignment made no progress")
        remaining = remaining[assignment.index_select(0, remaining) < 0]

    if capacity.any() or (assignment < 0).any():
        raise RuntimeError(
            f"incomplete balanced assignment; remaining capacity={capacity.tolist()}"
        )
    return assignment


def _full_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        return x.full_tensor().detach().clone()
    return x.detach().clone()


def _copy_from_full_(x: torch.Tensor, full_x: torch.Tensor) -> None:
    if not isinstance(x, DTensor):
        x.copy_(full_x)
        return
    distributed_x = distribute_like(x, full_x)
    get_local_tensor(x).copy_(get_local_tensor(distributed_x))


@torch.no_grad()
def _reorganize_expert_neurons_by_router_(m: "MoEBase", *, block_idx: int) -> dict[str, float]:
    """
    Reorganize normally initialized SwiGLU neurons around initialized router rows.

    Each neuron is moved atomically as its ``(w1, w3, w2)`` triplet. Therefore
    this operation is a pure permutation of initialized values: it changes only
    which expert owns each neuron.
    """
    from ..moe import DroplessMoEMLP, MoELinearRouter

    if not isinstance(m.router, MoELinearRouter):
        raise ValueError("router-centroid initialization requires a linear router")
    mlp = m.experts.mlp
    if not isinstance(mlp, DroplessMoEMLP):
        raise ValueError("router-centroid initialization requires a dropless MoE")

    w1 = _full_tensor(mlp.w1)
    w2 = _full_tensor(mlp.w2)
    w3 = _full_tensor(mlp.w3)
    num_experts = mlp.num_experts
    rows_per_expert = mlp.hidden_size
    num_rows = num_experts * rows_per_expert
    router = _full_tensor(m.router.weight).float().view(num_experts, mlp.d_model)

    w1_rows = w1.view(num_rows, mlp.d_model)
    w2_rows = w2.view(num_rows, mlp.d_model)
    w3_rows = w3.view(num_rows, mlp.d_model)
    router_hat = F.normalize(router, dim=-1)
    scores = 0.5 * (
        F.normalize(w1_rows.float(), dim=-1) @ router_hat.T
        + F.normalize(w3_rows.float(), dim=-1) @ router_hat.T
    )
    original_assignment = torch.arange(num_rows, device=scores.device) // rows_per_expert
    assignment = _balanced_fixed_centroid_assignment(scores, rows_per_expert)
    order = torch.argsort(assignment, stable=True)

    score_before = scores[torch.arange(num_rows, device=scores.device), original_assignment].mean()
    score_after = scores[torch.arange(num_rows, device=scores.device), assignment].mean()
    nearest_fraction = (assignment == scores.argmax(dim=1)).float().mean()

    _copy_from_full_(mlp.w1, w1_rows.index_select(0, order))
    _copy_from_full_(mlp.w2, w2_rows.index_select(0, order))
    _copy_from_full_(mlp.w3, w3_rows.index_select(0, order))

    counts = assignment.bincount(minlength=num_experts)
    diagnostics = {
        "score_before": score_before.item(),
        "score_after": score_after.item(),
        "nearest_fraction": nearest_fraction.item(),
        "min_cluster_size": float(counts.min().item()),
        "max_cluster_size": float(counts.max().item()),
    }
    log.info(
        "router-centroid expert init layer=%d score=%.6f->%.6f "
        "nearest_fraction=%.6f cluster_size=%d..%d pure_permutation=true",
        block_idx,
        diagnostics["score_before"],
        diagnostics["score_after"],
        diagnostics["nearest_fraction"],
        int(diagnostics["min_cluster_size"]),
        int(diagnostics["max_cluster_size"]),
    )
    return diagnostics


def init_linear(
    m: nn.Linear | nn.Conv1d, *, std: float = 0.02, generator: Optional[torch.Generator] = None
):
    _apply_init(
        nn.init.trunc_normal_,
        m.weight,
        mean=0.0,
        std=std,
        a=-3 * std,
        b=3 * std,
        generator=generator,
    )
    if m.bias is not None:
        nn.init.zeros_(m.bias)


class InitMethod(StrEnum):
    normal = "normal"
    """
    Every linear and embedding layer and initialized from a truncated normal distributed
    with standard deviation 0.02.
    """

    normalized = "normalized"
    """
    Follow the nGPT initialization scheme.
    """

    llama = "llama"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the number of layers.
    """

    llama_depth = "llama_depth"
    """
    Like :data:`normal`, but "output" layers are initialized with a standard deviation that's
    dependent on either ``d_model`` or the layer index.
    """

    fan_in = "fan_in"
    """
    Per-layer fan-in initialization where each weight matrix is initialized with
    ``std = 1/√d_in`` where ``d_in`` is the fan-in (number of input features) of that
    specific layer. Embeddings use ``std = 1.0`` with normal distribution.
    This provides forward-pass variance-preserving initialization adapted to each layer's
    specific dimensions, with no depth scaling.
    """

    def init_embeddings(
        self,
        m: nn.Embedding,
        *,
        d_model: int,
        embed_scale: Optional[float] = None,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (InitMethod.llama, InitMethod.llama_depth):
            _apply_init(nn.init.normal_, m.weight, generator=generator)
        elif self == InitMethod.normalized:
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=d_model**-0.5)
        elif self == InitMethod.fan_in:
            # Fan-in init uses std = 1.0 for embeddings, scaled down by embed_scale if set
            emb_std = 1.0 / embed_scale if embed_scale is not None else 1.0
            _apply_init(nn.init.normal_, m.weight, generator=generator, std=emb_std)
        else:
            _apply_init(
                nn.init.trunc_normal_,
                m.weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )

    def init_final_w_out(
        self,
        m: nn.Linear,
        *,
        d_model: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        if self in (
            InitMethod.llama,
            InitMethod.llama_depth,
            InitMethod.normalized,
            InitMethod.fan_in,
        ):
            std = d_model**-0.5
        init_linear(m, std=std, generator=generator)

    def init_attention(
        self,
        m: "SequenceMixer",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        m.init_weights(
            init_method=self,
            d_model=d_model,
            block_idx=block_idx,
            num_blocks=num_blocks,
            std=std,
            generator=generator,
        )

    def init_feed_forward(
        self,
        m: "FeedForward",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        # Compute std for w1 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w1 uses 1/√d_in where d_in = d_model (ignores base std parameter)
            std = m.w1.in_features**-0.5
        elif self == InitMethod.normalized:
            std = d_model**-0.5

        init_linear(m.w1, std=std, generator=generator)

        # Compute std for w3 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w3 uses 1/√d_in where d_in = d_model
            std = m.w3.in_features**-0.5
        elif self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5

        init_linear(m.w3, std=std, generator=generator)

        # Compute std for w2 initialization
        if self == InitMethod.fan_in:
            # For fan_in, w2 uses 1/√d_in where d_in = hidden_size
            std = m.w2.in_features**-0.5
        elif self == InitMethod.normalized:
            std = std / (2 * num_blocks) ** 0.5

        init_linear(m.w2, std=std, generator=generator)

    def init_feed_forward_moe(
        self,
        m: "MoEBase",
        *,
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ):
        from ..moe import DroplessMoEMLP, MoECentroidRouter, MoELinearRouter, MoEMLP

        if self == InitMethod.llama:
            std = std / (2 * num_blocks) ** 0.5
        elif self == InitMethod.llama_depth:
            std = std / (2 * (block_idx + 1)) ** 0.5
        elif self == InitMethod.fan_in:
            # For fan_in, router weight uses 1/√d_model
            std = d_model**-0.5

        if not isinstance(m.router, MoECentroidRouter):
            _apply_init(
                nn.init.trunc_normal_,
                cast(MoELinearRouter, m.router).weight,
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
                generator=generator,
            )

        mlp = cast(Union[MoEMLP, DroplessMoEMLP], m.experts.mlp)

        # Initialize w1 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w1,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w2 (maps hidden_size -> d_model, fan-in = hidden_size)
        if self == InitMethod.fan_in:
            std = mlp.hidden_size**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w2,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        # Initialize w3 (maps d_model -> hidden_size, fan-in = d_model)
        if self == InitMethod.fan_in:
            std = mlp.d_model**-0.5

        _apply_init(
            nn.init.trunc_normal_,
            mlp.w3,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=generator,
        )

        if m.reorganize_expert_init_by_router:
            _reorganize_expert_neurons_by_router_(m, block_idx=block_idx)
