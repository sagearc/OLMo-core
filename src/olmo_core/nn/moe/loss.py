from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard

from olmo_core.config import StrEnum
from olmo_core.distributed.utils import get_local_tensor


class MoELoadBalancingLossGranularity(StrEnum):
    """
    Defines the granularity for the router's load balancing loss.
    """

    local_batch = "local_batch"
    """
    The loss is always computed over the rank-local shard of the batch, ignoring any
    parallelism strategies used. This is ideal for minimizing the number of dropped tokens for
    any parallel strategy.
    """

    instance = "instance"
    """
    The loss is computed over each instance, taking into account any parallelism strategies used.
    """


def load_balancing_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batch_size_per_expert: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    granularity: MoELoadBalancingLossGranularity,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    expert_scores, batch_size_per_expert, batched_batch_size_per_expert = (
        get_local_tensor(expert_scores),
        get_local_tensor(batch_size_per_expert),
        get_local_tensor(batched_batch_size_per_expert),
    )

    B, S, _ = expert_scores.shape

    loss: torch.Tensor
    if granularity == MoELoadBalancingLossGranularity.instance:
        # shape: (B, num_experts)
        batched_batch_size_per_expert = batched_batch_size_per_expert.type_as(expert_scores)

        # NOTE: for CP it suffices to reduce the 'batched_batch_size_per_expert' across the CP group
        # and do the rest of the computation locally.
        if cp_mesh is not None:
            dist.all_reduce(batched_batch_size_per_expert, group=cp_mesh.get_group())

        # NOTE: for TP, the end result needs to be a DTensor over the TP mesh, so we handle this case
        # a little differently.
        if tp_mesh is not None:
            # NOTE: assumes sharded on sequence dimension and equal splits across TP group.
            dist.all_reduce(batched_batch_size_per_expert, group=tp_mesh.get_group())
            batched_batch_size_per_expert = DTensor.from_local(
                batched_batch_size_per_expert, tp_mesh, (Replicate(),)
            )
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, 1, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1, keepdim=True)
            # shape: (B, 1, num_experts) -> (B, num_experts)
            expert_scores = DTensor.from_local(expert_scores, tp_mesh, (Shard(1),)).mean(dim=1)
        else:
            # shape: (B * S, num_experts) -> (B, S, num_experts,) -> (B, num_experts)
            expert_scores = expert_scores.view(B, -1, num_experts).mean(dim=1)

        # We compute this across the TP and CP groups, so the 'loss_div_factor' should represent
        # the total number of tokens across the TP and CP groups.
        if loss_div_factor is None:
            # this gives us total number of tokens across TP + CP groups.
            loss_div_factor = batched_batch_size_per_expert.sum() / top_k

        # shape: scalar
        loss = (expert_scores * batched_batch_size_per_expert).sum() / loss_div_factor
    elif granularity == MoELoadBalancingLossGranularity.local_batch:
        # NOTE: We essentially ignore CP for this granularity, and for TP we still compute the loss
        # locally, but wrap as a DTensor and reduce it at the end because the end result has to be
        # a DTensor over the TP mesh.
        # Due to that DTensor reduction, with TP the 'loss_div_factor' should be the total number
        # of tokens across the TP group, but not the CP group.
        if loss_div_factor is None:
            loss_div_factor = B * S
            if tp_mesh is not None:
                loss_div_factor = loss_div_factor * tp_mesh.size()
        elif cp_mesh is not None:
            loss_div_factor = loss_div_factor / cp_mesh.size()

        # shape: (num_experts,)
        batch_size_per_expert = batch_size_per_expert.type_as(expert_scores)
        # shape: (B, S, num_experts) -> (B * S, num_experts)
        expert_scores = expert_scores.view(-1, num_experts)
        # shape: (B * S, num_experts) -> (num_experts,)
        expert_scores = expert_scores.mean(dim=0)
        # shape: scalar
        loss = torch.dot(batch_size_per_expert, expert_scores) / loss_div_factor
        if tp_mesh is not None:
            loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()
    else:
        raise NotImplementedError(granularity)

    scale = num_experts / top_k

    return scale * loss


def router_z_loss(
    *,
    expert_logits: torch.Tensor,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    tp_mesh: Optional[dist.DeviceMesh] = None,
    cp_mesh: Optional[dist.DeviceMesh] = None,
) -> torch.Tensor:
    expert_logits = get_local_tensor(expert_logits)
    B, S, _ = expert_logits.shape

    # NOTE: with TP, end result has to be a DTensor over the TP mesh, so we wrap as a DTensor
    # and reduce it. Due to this reduction, the 'loss_div_factor' should represent the total
    # number of tokens across the TP group (but not the CP group).
    if loss_div_factor is None:
        loss_div_factor = B * S
        if tp_mesh is not None:
            loss_div_factor = loss_div_factor * tp_mesh.size()
    elif cp_mesh is not None:
        loss_div_factor = loss_div_factor / cp_mesh.size()

    loss = torch.logsumexp(expert_logits, dim=-1).square().sum() / loss_div_factor
    if tp_mesh is not None:
        loss = DTensor.from_local(loss.unsqueeze(0), tp_mesh, (Shard(0),)).sum()

    return loss


def deepseek_seq_aux_loss(
    *,
    num_experts: int,
    top_k: int,
    expert_scores: torch.Tensor,
    batched_batch_size_per_expert: torch.Tensor,
    loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """
    DeepSeek-v3 complementary sequence-wise auxiliary loss (arXiv:2412.19437 §2.1.2):
    ``L = α · N_r · Σ_i f_i · P_i`` where ``f_i = (1/(K·S)) · #{tokens routed to i}`` and
    ``P_i = (1/S) Σ_t s'_{i,t}`` with ``s'`` = the L1-normalized per-token score —
    computed **per sequence**, then averaged across the batch.

    Matches Megatron-Core's ``_apply_seq_aux_loss``
    (megatron/core/transformer/moe/router.py:283-305): each instance in the batch is
    treated as one "sequence".

    Microbatch handling: when ``loss_div_factor`` is provided (= total batch tokens),
    the local microbatch contribution is normalized by global sequence count so that
    summing across microbatches yields the true per-batch sequence average. Without
    this, with M microbatches each of B_mb sequences, naive ``mean()`` per microbatch
    summed across the loop gives an M× over-weighted loss.

    The L1 normalization of ``s_t`` is required by the paper: ``s'_t`` is the gate-weight
    *distribution* (sums to 1 over experts), not the raw sigmoid output. Renormalizing
    here makes this function self-contained for both softmax (already sums to 1, no-op)
    and sigmoid (each ``s_i`` is independent, ``Σ_i s_i`` ≠ 1) inputs.

    :param num_experts: Total number of experts (``N_r`` in the paper).
    :param top_k: Number of experts selected per token (``K_r`` in the paper).
    :param expert_scores: Per-expert scores after the gating function,
        shape ``(B, S, num_experts)``. Will be L1-normalized internally.
    :param batched_batch_size_per_expert: Per-instance counts of tokens routed to each
        expert, shape ``(B, num_experts)``. Must be detached.
    :param loss_div_factor: Total tokens in the full (multi-microbatch) batch.
        If provided, used to compute total batch sequences = ``loss_div_factor / S``
        and divides the local sum so cross-microbatch summation yields the per-batch
        average. If None, falls back to mean over local sequences only.
    """
    expert_scores = get_local_tensor(expert_scores)
    batched_batch_size_per_expert = get_local_tensor(batched_batch_size_per_expert)
    B, S, _ = expert_scores.shape

    # L1-normalize per token to get the s'_t distribution from the paper.
    normalized_scores = expert_scores / expert_scores.sum(dim=-1, keepdim=True).clamp(min=1e-20)

    # Per-sequence f_i: shape (B, num_experts), sums to 1 across experts within each row.
    f_i = batched_batch_size_per_expert.float() / float(S * top_k)
    # Per-sequence P_i: mean L1-normalized score per expert within each instance.
    p_i = normalized_scores.mean(dim=1)  # (B, num_experts)

    # Per-sequence loss; sum then normalize so cross-microbatch sum gives the per-batch
    # average (loss_div_factor / S = total sequences in the full batch).
    per_seq_loss = float(num_experts) * (f_i * p_i).sum(dim=-1)  # (B,)
    if loss_div_factor is None:
        return per_seq_loss.mean()
    total_seqs = loss_div_factor / float(S)
    return per_seq_loss.sum() / total_seqs
