import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union, cast

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

import olmo_core.ops.moe as ops
from olmo_core.config import DType, StrEnum
from olmo_core.distributed.utils import (
    _HiddenTensor,
    distribute_like,
    get_local_tensor,
    hide_from_torch,
    is_distributed,
    unhide_from_torch,
)
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.utils import get_default_device

from ..config import ModuleConfig
from .loss import (
    MoELoadBalancingLossGranularity,
    deepseek_seq_aux_loss,
    load_balancing_loss,
    router_z_loss,
)

if TYPE_CHECKING:
    from olmo_core.train.common import ReduceType

__all__ = [
    "MoERouter",
    "MoELinearRouter",
    "MoERouterConfig",
    "MoERouterType",
    "MoERouterGatingFunction",
]


log = logging.getLogger(__name__)


# NOTE: To enable end-to-end benchmarking without convergence we
# support a flag to force the router to assign items/tokens uniformly
# across the experts. We do this with a custom autograd operation
# so that PyTorch still executes the full set of router operation.
class _UniformExpertAssignment(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, num_experts: int):
        del ctx
        out = torch.arange(x.numel(), dtype=x.dtype, device=x.device)
        out = torch.remainder(out, num_experts)
        return out.view(x.shape)


_uniform_expert_assignment: Callable[
    [torch.Tensor, int], torch.Tensor
] = _UniformExpertAssignment.apply  # type: ignore


class MoERouterType(StrEnum):
    """
    An enumeration of the different MoE router implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoELinearRouter`
    """


class MoERouterGatingFunction(StrEnum):
    softmax = "softmax"
    sigmoid = "sigmoid"


@dataclass
class MoERouterConfig(ModuleConfig):
    """
    A configuration class for easily building any of the different MoE router modules.
    """

    name: MoERouterType = MoERouterType.default
    """
    The name of the implementation.
    """
    top_k: int = 1
    jitter_eps: Optional[float] = None
    normalize_expert_weights: Optional[float] = None
    uniform_expert_assignment: bool = False
    bias_gamma: Optional[float] = None
    gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax
    seq_aux_loss_weight: Optional[float] = None
    """
    If set, enables the DeepSeek-v3 complementary sequence-wise auxiliary loss
    (arXiv:2412.19437 §2.1.2) with this weight. Paper value: ``1e-4``. Computed as a
    microbatch-wise approximation; see :func:`olmo_core.nn.moe.loss.deepseek_seq_aux_loss`.
    """
    ema_zscore_normalize: bool = False
    """
    If True, z-normalize router logits per expert using an EMA of mean / squared-mean
    before applying the gating function. Composes with both ``softmax`` and ``sigmoid``.
    """
    ema_zscore_alpha: float = 0.99
    """
    EMA decay used when ``ema_zscore_normalize`` is enabled.
    """
    dtype: Optional[DType] = None

    def num_params(self, d_model: int, num_experts: int) -> int:
        """
        The number of params that the module will have once built.

        :param d_model: The model dimensionality.
        """
        num_params = 0
        if self.name == MoERouterType.default:
            num_params += d_model * num_experts
        else:
            raise NotImplementedError

        return num_params

    def build(
        self,
        d_model: int,
        num_experts,
        *,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
        init_device: str = "cpu",
    ) -> "MoERouter":
        """
        Build the corresponding MoE router module.

        :param d_model: The model dimensionality.
        :param num_experts: The number of experts.
        :param init_device: The device initialize the parameters on, e.g. "cpu", "meta".
        """
        kwargs = self.as_dict(exclude_none=True, recurse=False)
        kwargs.pop("name")
        kwargs.update(
            d_model=d_model,
            num_experts=num_experts,
            init_device=init_device,
            lb_loss_weight=lb_loss_weight,
            lb_loss_granularity=lb_loss_granularity,
            z_loss_weight=z_loss_weight,
        )
        if self.dtype is not None:
            kwargs["dtype"] = self.dtype.as_pt()
        elif dtype is not None:
            kwargs["dtype"] = dtype

        try:
            if self.name == MoERouterType.default:
                return MoELinearRouter(**kwargs)
            else:
                raise NotImplementedError(self.name)
        except TypeError as e:
            raise OLMoConfigurationError(
                f"invalid options for '{self.name}' {self.__class__.__name__}, {e}"
            ) from e


class MoERouter(nn.Module):
    """
    A base class for MoE router modules.

    :param d_model: The model dimensionality (hidden size).
    :param num_experts: The total number of experts.
    :param top_k: The number of experts to assign to each item/token.
    :param jitter_eps: Controls the amount of noise added to the input during training.
    :param normalize_expert_weights: The type of norm (e.g. ``2.0`` for L2 norm) to use to normalize
        the expert weights.
    :param uniform_expert_assignment: Force uniform assignment. Useful for benchmarking.
    :param bias_gamma: If set to a positive float, experts scores for top-k routing will be adjusted
        by a bias following the "auxiliary-loss-free load balancing" strategy from DeepSeek-v3.
        A reasonable value is on the order of 0.0001.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_experts: int,
        top_k: int = 1,
        jitter_eps: Optional[float] = None,
        normalize_expert_weights: Optional[float] = None,
        uniform_expert_assignment: bool = False,
        bias_gamma: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        seq_aux_loss_weight: Optional[float] = None,
        ema_zscore_normalize: bool = False,
        ema_zscore_alpha: float = 0.99,
        lb_loss_weight: Optional[float] = None,
        lb_loss_granularity: MoELoadBalancingLossGranularity = MoELoadBalancingLossGranularity.local_batch,
        z_loss_weight: Optional[float] = None,
        init_device: str = "cpu",
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_eps = jitter_eps
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment
        self.bias_gamma = bias_gamma
        self.gating_function = gating_function
        self.seq_aux_loss_weight = seq_aux_loss_weight
        self.ema_zscore_normalize = ema_zscore_normalize
        if ema_zscore_normalize:
            assert 0.0 < ema_zscore_alpha < 1.0, (
                f"ema_zscore_alpha must be in (0, 1), got {ema_zscore_alpha}; "
                "the bias-correction term `1 - α^t` is undefined at the boundaries."
            )
        self.ema_zscore_alpha = ema_zscore_alpha
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[dist.DeviceMesh] = None
        self.tp_mesh: Optional[dist.DeviceMesh] = None

        if self.bias_gamma is not None:
            assert self.bias_gamma > 0
            self.register_buffer("score_bias", torch.zeros(self.num_experts, device=init_device))
        else:
            self.register_buffer("score_bias", None)

        # EMA state — registered as buffers so they get checkpointed by the sharded
        # checkpointer (mirrors `score_bias`). Buffer registration is safe here because
        # the EMA is only *read* in forward; updates happen in `post_batch`, outside any
        # torch.compile traced region. Initialized to zero so Adam-style bias correction
        # (1 / (1 - α^t)) yields unbiased mean / E[X²] estimates from step 1 onwards.
        if self.ema_zscore_normalize:
            self.register_buffer(
                "_ema_mean",
                torch.zeros(self.num_experts, dtype=torch.float32, device=init_device),
            )
            self.register_buffer(
                "_ema_sq",
                torch.zeros(self.num_experts, dtype=torch.float32, device=init_device),
            )
            self.register_buffer(
                "_ema_step_count",
                torch.zeros((), dtype=torch.long, device=init_device),
            )
        else:
            self.register_buffer("_ema_mean", None)
            self.register_buffer("_ema_sq", None)
            self.register_buffer("_ema_step_count", None)

        # NOTE: we don't use buffers for these because we don't want FSDP to manage them, and we
        # don't use a BufferCache because `torch.compile()` doesn't handle that well when we're modifying
        # values in the cache.
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=init_device)
        )
        self._score_bias_batch_size_per_expert: Optional[_HiddenTensor] = None
        self._load_balancing_loss: Optional[_HiddenTensor] = None
        self._z_loss: Optional[_HiddenTensor] = None
        self._seq_aux_loss: Optional[_HiddenTensor] = None
        # Per-step EMA accumulators (raw sums, not means). Reset in `post_batch` after each
        # optimizer step. Hidden from torch so FSDP doesn't try to manage them.
        self._ema_logit_sum_accum: Optional[_HiddenTensor] = None
        self._ema_logit_sq_sum_accum: Optional[_HiddenTensor] = None
        self._ema_token_count_accum: int = 0

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )

        if self.bias_gamma is not None:
            assert self.score_bias is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            score_bias.zero_()
            self._score_bias_batch_size_per_expert = hide_from_torch(
                torch.zeros(self.num_experts, device=self.device)
            )

        if self.lb_loss_weight is not None:
            self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.z_loss_weight is not None:
            self._z_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.seq_aux_loss_weight is not None:
            self._seq_aux_loss = hide_from_torch(torch.zeros([], device=self.device))

        if self.ema_zscore_normalize:
            assert self._ema_mean is not None and self._ema_sq is not None
            assert self._ema_step_count is not None
            cast(torch.Tensor, self._ema_mean).zero_()
            cast(torch.Tensor, self._ema_sq).zero_()
            cast(torch.Tensor, self._ema_step_count).zero_()
            self._ema_logit_sum_accum = None
            self._ema_logit_sq_sum_accum = None
            self._ema_token_count_accum = 0

    @property
    def device(self) -> torch.device:
        return get_default_device()

    @property
    def score_bias_batch_size_per_expert(self) -> Optional[torch.Tensor]:
        if self.bias_gamma is not None:
            if self._score_bias_batch_size_per_expert is None:
                self._score_bias_batch_size_per_expert = hide_from_torch(
                    torch.zeros(self.num_experts, device=self.device)
                )
            elif self._score_bias_batch_size_per_expert.device != self.device:
                self._score_bias_batch_size_per_expert = self._score_bias_batch_size_per_expert.to(
                    self.device
                )
        return (
            None
            if self._score_bias_batch_size_per_expert is None
            else unhide_from_torch(self._score_bias_batch_size_per_expert)
        )

    @score_bias_batch_size_per_expert.setter
    def score_bias_batch_size_per_expert(self, value: torch.Tensor):
        self._score_bias_batch_size_per_expert = hide_from_torch(value)

    @property
    def batch_size_per_expert(self) -> torch.Tensor:
        if self._batch_size_per_expert.device != self.device:
            self._batch_size_per_expert = self._batch_size_per_expert.to(self.device)
        return unhide_from_torch(self._batch_size_per_expert)

    @batch_size_per_expert.setter
    def batch_size_per_expert(self, value: torch.Tensor):
        self._batch_size_per_expert = hide_from_torch(value)

    @property
    def load_balancing_loss(self) -> Optional[torch.Tensor]:
        if self.lb_loss_weight is not None:
            if self._load_balancing_loss is None:
                self._load_balancing_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._load_balancing_loss.device != self.device:
                self._load_balancing_loss = self._load_balancing_loss.to(self.device)
        return (
            None
            if self._load_balancing_loss is None
            else unhide_from_torch(self._load_balancing_loss)
        )

    @load_balancing_loss.setter
    def load_balancing_loss(self, value: torch.Tensor):
        self._load_balancing_loss = hide_from_torch(value)

    @property
    def z_loss(self) -> Optional[torch.Tensor]:
        if self.z_loss_weight is not None:
            if self._z_loss is None:
                self._z_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._z_loss.device != self.device:
                self._z_loss = self._z_loss.to(self.device)
        return None if self._z_loss is None else unhide_from_torch(self._z_loss)

    @z_loss.setter
    def z_loss(self, value: torch.Tensor):
        self._z_loss = hide_from_torch(value)

    @property
    def seq_aux_loss(self) -> Optional[torch.Tensor]:
        if self.seq_aux_loss_weight is not None:
            if self._seq_aux_loss is None:
                self._seq_aux_loss = hide_from_torch(torch.zeros([], device=self.device))
            elif self._seq_aux_loss.device != self.device:
                self._seq_aux_loss = self._seq_aux_loss.to(self.device)
        return None if self._seq_aux_loss is None else unhide_from_torch(self._seq_aux_loss)

    @seq_aux_loss.setter
    def seq_aux_loss(self, value: torch.Tensor):
        self._seq_aux_loss = hide_from_torch(value)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False):
        if not self.training:
            return

        # ---- DeepSeek bias-rule update --------------------------------------------------
        if self.bias_gamma is not None:
            assert self.score_bias is not None
            assert self.score_bias_batch_size_per_expert is not None
            score_bias = cast(torch.Tensor, self.score_bias)
            batch_size_per_expert = self.score_bias_batch_size_per_expert

            # Maybe reduce across the process group.
            if is_distributed():
                dist.all_reduce(batch_size_per_expert, group=self.group)

            ideal_batch_size_per_expert = batch_size_per_expert.mean(
                dim=0, keepdim=True, dtype=torch.float32
            )
            bias_delta = (
                self.bias_gamma * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
            )
            # NOTE: have to be careful here to manage the case where `score_bias` is a DTensor.
            bias_delta = distribute_like(score_bias, bias_delta)

            if not dry_run:
                get_local_tensor(score_bias).add_(get_local_tensor(bias_delta))

            # Reset the accumulator.
            batch_size_per_expert.zero_()

        # ---- EMA z-score per-step update ------------------------------------------------
        if self.ema_zscore_normalize and self._ema_logit_sum_accum is not None:
            assert self._ema_mean is not None and self._ema_sq is not None
            assert self._ema_step_count is not None
            assert self._ema_logit_sq_sum_accum is not None
            ema_mean = cast(torch.Tensor, self._ema_mean)
            ema_sq = cast(torch.Tensor, self._ema_sq)
            ema_step = cast(torch.Tensor, self._ema_step_count)
            local_sum = unhide_from_torch(self._ema_logit_sum_accum)
            local_sq = unhide_from_torch(self._ema_logit_sq_sum_accum)
            local_count = torch.tensor(
                float(self._ema_token_count_accum), dtype=torch.float32, device=local_sum.device
            )

            # Reduce SUMS and COUNT (not means) to get exact global mean under uneven
            # per-rank token counts.
            if is_distributed():
                dist.all_reduce(local_sum, group=self.group)
                dist.all_reduce(local_sq, group=self.group)
                dist.all_reduce(local_count, group=self.group)

            global_mean = local_sum / local_count
            global_sq = local_sq / local_count

            if not dry_run:
                alpha = self.ema_zscore_alpha
                ema_mean.mul_(alpha).add_(global_mean, alpha=1.0 - alpha)
                ema_sq.mul_(alpha).add_(global_sq, alpha=1.0 - alpha)
                ema_step.add_(1)  # tracks t for Adam-style bias correction in forward

            # Reset accumulators whether or not we applied the update — never leak.
            self._ema_logit_sum_accum = None
            self._ema_logit_sq_sum_accum = None
            self._ema_token_count_accum = 0

    def _apply_ema_zscore(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Z-normalize ``logits`` per expert using the EMA snapshot from the previous optimizer
        step, then accumulate raw stats (sum, sum-of-squares, token count) so that
        :meth:`post_batch` can apply a single EMA update per step. Gradient flows through
        ``logits`` only — EMA buffers and accumulators are detached.

        The EMA snapshot is constant across all microbatches within a step, so all
        microbatches see identical ``μ``, ``σ``. This makes the per-step EMA decay
        unambiguous (one ``α``-update per optimizer step, not per microbatch).

        Adam-style bias correction (``/ (1 - α^t)``) is applied so that the EMA gives
        unbiased mean / ``E[X²]`` estimates from step 1 onwards, instead of taking
        ~``1/(1-α)`` steps to ramp up from the zero init.

        Step ``t = 0`` (no stats yet) returns ``logits`` unchanged — the very first
        forward sees raw logits, which is acceptable for one batch and avoids
        normalizing by an undefined std.
        """
        assert self._ema_mean is not None and self._ema_sq is not None
        assert self._ema_step_count is not None
        ema_mean = cast(torch.Tensor, self._ema_mean)
        ema_sq = cast(torch.Tensor, self._ema_sq)
        step = int(cast(torch.Tensor, self._ema_step_count).item())
        num_experts = logits.shape[-1]

        if step == 0:
            normalized = logits
        else:
            bc = 1.0 - (self.ema_zscore_alpha**step)
            mean_hat = ema_mean / bc
            sq_hat = ema_sq / bc
            # Std floor of 1e-2 (clamp on var = 1e-4) keeps `(logits - μ) / σ` from
            # blowing up to Inf under bf16 if the EMA variance ever collapses (e.g.
            # immediately post-checkpoint-load before stats refill, or transient
            # collapse during init).
            ema_std = (sq_hat - mean_hat.pow(2)).clamp(min=1e-4).sqrt()
            normalized = (logits - mean_hat) / ema_std

        # `torch.is_grad_enabled()` matches the guard the existing aux-loss path uses at
        # forward(): under activation checkpointing, the forward call inside the no_grad
        # region must NOT mutate state — only the recomputation during backward should.
        # olmo-core currently raises if you try to AC-wrap an MoE block (model.py:727),
        # but this guard keeps EMA correct if that protection is ever relaxed.
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                flat = logits.detach().view(-1, num_experts).float()
                # Accumulate sums (not means) and the token count separately, so that
                # `post_batch` can compute the exact global mean across microbatches AND
                # ranks via SUM-and-COUNT all-reduce — correct under uneven per-rank
                # token counts (e.g. variable padding), unlike a mean-of-means reduction.
                batch_sum = flat.sum(dim=0)
                batch_sq_sum = flat.pow(2).sum(dim=0)
                n_tokens = flat.shape[0]
                if self._ema_logit_sum_accum is None:
                    self._ema_logit_sum_accum = hide_from_torch(batch_sum.clone())
                    self._ema_logit_sq_sum_accum = hide_from_torch(batch_sq_sum.clone())
                    self._ema_token_count_accum = n_tokens
                else:
                    unhide_from_torch(self._ema_logit_sum_accum).add_(batch_sum)
                    assert self._ema_logit_sq_sum_accum is not None
                    unhide_from_torch(self._ema_logit_sq_sum_accum).add_(batch_sq_sum)
                    self._ema_token_count_accum += n_tokens

        return normalized

    def jitter(self, x: torch.Tensor) -> torch.Tensor:
        if self.jitter_eps is None or not self.training:
            return x
        else:
            low = 1.0 - self.jitter_eps
            high = 1.0 + self.jitter_eps
            noise = torch.rand_like(x)
            return x * (low + noise * (high - low))

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights: torch.Tensor
        expert_indices: torch.Tensor
        if self.bias_gamma is None:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0), self.top_k, dim=-1  # type: ignore
                )
            expert_weights = scores.gather(-1, expert_indices)

        if self.uniform_expert_assignment:
            expert_indices = _uniform_expert_assignment(expert_indices, self.num_experts)
            expert_weights = scores.gather(-1, expert_indices)

        return expert_weights, expert_indices

    @abstractmethod
    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given the input ``x`` of shape ``(*, d_model)``, compute the un-normalized expert scores.

        :returns: The expert logits, shape ``(*, num_experts)``.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out: Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]] = {}

        # Load imbalance.
        batch_size_per_expert = self.batch_size_per_expert
        out["load imbalance"] = (
            batch_size_per_expert.max() / batch_size_per_expert.mean(dtype=torch.float),
            ReduceType.max,
        )

        # Load balancing loss.
        if self.lb_loss_weight is not None:
            assert self.load_balancing_loss is not None
            out["load balancing loss"] = (
                self.lb_loss_weight * self.load_balancing_loss,
                ReduceType.mean,
            )
            out["load balancing loss unscaled"] = (
                self.load_balancing_loss.clone(),
                ReduceType.mean,
            )

        # Router Z loss.
        if self.z_loss_weight is not None:
            assert self.z_loss is not None
            out["router Z loss"] = (self.z_loss_weight * self.z_loss, ReduceType.mean)
            out["router Z loss unscaled"] = (self.z_loss.clone(), ReduceType.mean)

        # DeepSeek sequence-wise aux loss.
        if self.seq_aux_loss_weight is not None:
            assert self.seq_aux_loss is not None
            out["seq aux loss"] = (
                self.seq_aux_loss_weight * self.seq_aux_loss,
                ReduceType.mean,
            )
            out["seq aux loss unscaled"] = (self.seq_aux_loss.clone(), ReduceType.mean)

        if reset:
            self.reset_metrics()

        return out

    def reset_metrics(self):
        if (bz_per_expert := self.batch_size_per_expert) is not None:
            bz_per_expert.zero_()
        if (lb_loss := self.load_balancing_loss) is not None:
            lb_loss.zero_()
        if (z_loss := self.z_loss) is not None:
            z_loss.zero_()
        if (seq_aux := self.seq_aux_loss) is not None:
            seq_aux.zero_()

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Given the input ``x`` of shape ``(B, S, d_model)``, compute the experts assignment.

        :returns: The expert weights of shape ``(B, S, top_k)``,
            the expert indices of shape ``(B, S, top_k)``,
            the total number of items routed to each expert, with shape ``(num_experts,)``,
            and optionally the auxiliary losses.
        """
        # shape: (batch_size, seq_len, d_model)
        x = self.jitter(x)

        # shape: (batch_size, seq_len, num_experts)
        logits = self.get_expert_logits(x).float()

        # Optionally z-normalize logits per expert using EMA stats. Composes with both
        # softmax and sigmoid below. Gradient flows through `logits` (not the EMA stats).
        gating_logits = logits
        if self.ema_zscore_normalize:
            gating_logits = self._apply_ema_zscore(logits)

        # shape: (batch_size, seq_len, num_experts)
        if self.gating_function == MoERouterGatingFunction.softmax:
            scores = gating_logits.softmax(dim=-1)
        elif self.gating_function == MoERouterGatingFunction.sigmoid:
            scores = F.sigmoid(gating_logits) + 1e-7
        else:
            raise NotImplementedError(self.gating_function)

        # shape: (batch_size, seq_len, top_k)
        expert_weights, expert_indices = self.get_top_k(scores)

        if self.normalize_expert_weights is not None:
            expert_weights = expert_weights.div(
                torch.norm(
                    expert_weights,
                    p=self.normalize_expert_weights,
                    dim=-1,
                    keepdim=True,
                )
            )

        with torch.no_grad():
            # Histogram the expert ids to identify the number of items/tokens routed to each expert.
            # shape: (batch_size, seq_len, num_experts)
            batched_batch_size_per_expert = ops.batched_histc(expert_indices, self.num_experts)
            # shape: (batch_size, num_experts)
            batched_batch_size_per_expert = batched_batch_size_per_expert.sum(dim=1)
            # shape: (num_experts,)
            batch_size_per_expert = batched_batch_size_per_expert.sum(dim=0)

        # Maybe compute auxiliary losses and accumulate metrics.
        aux_loss: Optional[torch.Tensor] = None
        if self.training and torch.is_grad_enabled():
            with torch.autocast(enabled=False, device_type=x.device.type):
                if self.lb_loss_weight is not None:
                    assert self.load_balancing_loss is not None

                    # Make sure scores are normalized, otherwise load balancing loss doesn't work well.
                    if self.gating_function == MoERouterGatingFunction.sigmoid:
                        scores = scores / scores.sum(dim=-1, keepdim=True)

                    lb_loss = load_balancing_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        batch_size_per_expert=batch_size_per_expert,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,
                        granularity=self.lb_loss_granularity,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    # Strip DTensor wrapper before accumulating into the local hidden scalar
                    # — under TP, lb_loss is a DTensor and `local += dtensor` either errors or
                    # silently triggers cross-rank averaging of what should be a local stat.
                    self.load_balancing_loss += get_local_tensor(lb_loss.detach())

                    scaled_lb_loss = self.lb_loss_weight * lb_loss
                    aux_loss = scaled_lb_loss

                if self.z_loss_weight is not None:
                    assert self.z_loss is not None

                    z_loss = router_z_loss(
                        expert_logits=logits,
                        loss_div_factor=loss_div_factor,
                        tp_mesh=self.tp_mesh,
                        cp_mesh=self.cp_mesh,
                    )
                    self.z_loss += get_local_tensor(z_loss.detach())

                    scaled_z_loss = self.z_loss_weight * z_loss
                    aux_loss = scaled_z_loss if aux_loss is None else aux_loss + scaled_z_loss

                if self.seq_aux_loss_weight is not None:
                    assert self.seq_aux_loss is not None

                    seq_aux = deepseek_seq_aux_loss(
                        num_experts=self.num_experts,
                        top_k=self.top_k,
                        expert_scores=scores,
                        batched_batch_size_per_expert=batched_batch_size_per_expert,
                        loss_div_factor=loss_div_factor,
                    )
                    self.seq_aux_loss += get_local_tensor(seq_aux.detach())

                    scaled_seq_aux = self.seq_aux_loss_weight * seq_aux
                    aux_loss = scaled_seq_aux if aux_loss is None else aux_loss + scaled_seq_aux

            self.batch_size_per_expert += batch_size_per_expert
            if self.bias_gamma is not None:
                assert self.score_bias_batch_size_per_expert is not None
                self.score_bias_batch_size_per_expert += batch_size_per_expert

        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        del float8_enabled
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(Shard(1),),
                desired_input_layouts=(Shard(1),),
                use_local_output=True,
            ),
        )
        self.tp_mesh = tp_mesh

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.cp_mesh = cp_mesh


class MoELinearRouter(MoERouter):
    """
    A simple, learned, linear router.
    """

    def __init__(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
        **kwargs,
    ):
        super().__init__(init_device=init_device, **kwargs)
        # NOTE: this parameter needs to have a large enough first dimension (which would be num experts)
        # in order to be sharded over big world sizes with FSDP. So we flatten it to a single dimension tensor.
        # And for that reason we don't support a 'bias' option.
        self.weight = nn.Parameter(
            torch.empty(self.num_experts * self.d_model, device=init_device, dtype=dtype)
        )
        self.reset_parameters()

    @property
    def device(self) -> torch.device:
        return self.weight.device if self.weight.device.type != "meta" else torch.device("cpu")

    def reset_parameters(self) -> None:
        super().reset_parameters()
        nn.init.trunc_normal_(self.weight, std=0.02, a=-3 * 0.02, b=3 * 0.02)

    def extra_repr(self):
        return f"in_features={self.d_model}, num_experts={self.num_experts}"

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x.float(), get_local_tensor(self.weight).view(self.num_experts, self.d_model).float()
        )

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        super().apply_tp(tp_mesh, float8_enabled=float8_enabled)
        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )
