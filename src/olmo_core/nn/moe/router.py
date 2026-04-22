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
    get_full_tensor,
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
    If True, z-normalize router logits per expert using EMA estimates of per-expert
    mean and variance before applying the gating function. Composes with both
    ``softmax`` and ``sigmoid``. The variance is tracked *directly* (via a
    sample-mean-based observation), not reconstructed from ``E[X²] − E[X]²``.
    See ``EMA_ZSCORE_ANALYSIS.md`` in this directory for the design rationale.
    """
    ema_zscore_alpha: float = 0.99
    """
    EMA decay used for the mean level and for the variance level.
    """
    ema_zscore_trend: bool = False
    """
    If True, apply Holt's linear-trend smoothing (undamped) to the **mean only**.
    Holt's tracks two state variables per expert:

    - *level* ``ℓ`` — the smoothed estimate of the per-expert mean logit right now.
    - *trend* ``b`` — the smoothed estimate of how fast that mean is drifting per
      optimizer step.

    Level update ``ℓ_t = α·(ℓ_{t-1}+b_{t-1}) + (1-α)·y_t`` blends the previous
    level-plus-trend forecast with the current batch's mean ``y_t``. Forecast at
    forward-time is ``μ̂ = ℓ+b``, which has zero steady-state lag for linear drift
    (vs. plain EMA's α/(1-α) ≈ 99-step lag at α=0.99). The variance is always
    smoothed with plain EMA — no trend — regardless of this flag. The asymmetry
    is intentional: mean-bias feeds back through routing and needs zero-lag
    tracking; variance-bias is second-order and plain EMA is strictly safer
    (cannot drive ``σ̂²`` below zero).
    """
    ema_zscore_trend_beta: float = 0.9
    """
    Old-weight for Holt's trend buffer (same convention as ``ema_zscore_alpha``):
    close to 1 = slow to change. Trend update is
    ``b_t = β·b_{t-1} + (1-β)·(ℓ_t - ℓ_{t-1})`` — an EMA over per-step changes in
    the level. At 0.9 the trend is 90% history + 10% new level-diff, giving a
    ~10-step memory (textbook Holt's β*=0.1). Lower → more responsive, noisier.
    Only used when ``ema_zscore_trend`` is enabled.
    """
    ema_zscore_trend_warmup: int = 1000
    """
    Hold the mean's trend buffer at zero (plain EMA + Adam-style bias correction)
    for the first ``ema_zscore_trend_warmup`` optimizer steps, then switch to Holt's
    forecast ``(ℓ + b) / (1 - α^t)``. Prevents the trend from fitting the non-linear
    weight trajectory during LR warmup and then forecasting phantom drift afterwards.
    Set to 0 to skip the plain-EMA phase (not recommended — the Holt level starts
    from zero init and produces biased forecasts for ~``1/(1-α)`` steps until the
    bias correction ``1 / (1 - α^t)`` converges to 1). Only used when
    ``ema_zscore_trend`` is enabled. Does not affect the variance, which is always
    plain EMA.
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
        ema_zscore_trend: bool = False,
        ema_zscore_trend_beta: float = 0.9,
        ema_zscore_trend_warmup: int = 1000,
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
        self.ema_zscore_trend = ema_zscore_trend
        if ema_zscore_trend:
            assert ema_zscore_normalize, (
                "ema_zscore_trend requires ema_zscore_normalize=True; it's a modifier "
                "on the EMA path, not a standalone mode."
            )
            assert (
                0.0 < ema_zscore_trend_beta < 1.0
            ), f"ema_zscore_trend_beta must be in (0, 1), got {ema_zscore_trend_beta}."
            assert (
                ema_zscore_trend_warmup >= 0
            ), f"ema_zscore_trend_warmup must be >= 0, got {ema_zscore_trend_warmup}."
        self.ema_zscore_trend_beta = ema_zscore_trend_beta
        self.ema_zscore_trend_warmup = ema_zscore_trend_warmup
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
        # checkpointer (mirrors `score_bias`). Buffer registration is safe here
        # because the EMA is only *read* in forward; updates happen in `post_batch`,
        # outside any torch.compile traced region. Initialized to zero so Adam-style
        # bias correction (1 / (1 - α^t)) yields unbiased mean / variance estimates
        # from step 1 onwards.
        #
        # - `_ema_mean`: per-expert running estimate of the mean logit (Holt's ``ℓ``
        #   when `ema_zscore_trend` is on, plain-EMA level otherwise).
        # - `_ema_var`: per-expert running estimate of Var(logit), computed from the
        #   sample-based observation `var_obs = sq_sum/count − (sum/count)²`. NOT
        #   `E[X²]` — the old `E[X²]` formulation created an algebraic cancellation
        #   with the mean-trend under sustained drift; see EMA_ZSCORE_ANALYSIS.md.
        # - `_ema_step_count`: optimizer-step counter `t` used in the bias-correction
        #   factor `1 - α^t`. Long-dtype so it never overflows.
        if self.ema_zscore_normalize:
            self.register_buffer(
                "_ema_mean",
                torch.zeros(self.num_experts, dtype=torch.float32, device=init_device),
            )
            self.register_buffer(
                "_ema_var",
                torch.zeros(self.num_experts, dtype=torch.float32, device=init_device),
            )
            self.register_buffer(
                "_ema_step_count",
                torch.zeros((), dtype=torch.long, device=init_device),
            )
        else:
            self.register_buffer("_ema_mean", None)
            self.register_buffer("_ema_var", None)
            self.register_buffer("_ema_step_count", None)

        # Holt's trend companion for the MEAN only (`b` in textbook notation) —
        # per-expert running estimate of the per-step drift of `_ema_mean`.
        # Initialized to zero; the level update reduces to plain EMA when trend ≈ 0,
        # so startup is graceful — no Adam-style bias correction needed for the
        # trend buffer itself. Variance has no trend buffer by design.
        if self.ema_zscore_trend:
            self.register_buffer(
                "_ema_mean_trend",
                torch.zeros(self.num_experts, dtype=torch.float32, device=init_device),
            )
        else:
            self.register_buffer("_ema_mean_trend", None)

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
        # Python-int shadow of `_ema_step_count` to avoid a device→host `.item()` sync on
        # the forward hot path (called every microbatch via `_apply_ema_zscore`). Kept in
        # lockstep with the buffer: incremented in `post_batch`, zeroed in
        # `reset_parameters`, and re-synced from the buffer in `_load_from_state_dict`
        # so checkpoint resume doesn't leave it stale.
        self._ema_step_py: int = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        """
        Rehydrate the Python shadow ``_ema_step_py`` from the checkpointed buffer after
        any state-dict load (full or sharded). Without this, the buffer would hold the
        resumed step count while the Python int stayed at 0, causing `_apply_ema_zscore`
        to apply the wrong bias correction ``1 / (1 - α^t)`` until the next post_batch
        re-synced them by accident.
        """
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.ema_zscore_normalize and self._ema_step_count is not None:
            self._ema_step_py = int(cast(torch.Tensor, self._ema_step_count).item())

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
            assert self._ema_mean is not None and self._ema_var is not None
            assert self._ema_step_count is not None
            cast(torch.Tensor, self._ema_mean).zero_()
            cast(torch.Tensor, self._ema_var).zero_()
            cast(torch.Tensor, self._ema_step_count).zero_()
            self._ema_step_py = 0
            self._ema_logit_sum_accum = None
            self._ema_logit_sq_sum_accum = None
            if self.ema_zscore_trend:
                assert self._ema_mean_trend is not None
                cast(torch.Tensor, self._ema_mean_trend).zero_()
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
            assert self._ema_mean is not None and self._ema_var is not None
            assert self._ema_step_count is not None
            assert self._ema_logit_sq_sum_accum is not None
            ema_mean = cast(torch.Tensor, self._ema_mean)
            ema_var = cast(torch.Tensor, self._ema_var)
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
            # Sample-mean-based variance observation: var_obs = (1/n) Σ (x_i − x̄)².
            # Unbiased of the forecast μ̂, so smoothing it cannot leak forecast error
            # into σ̂ (which was the failure mode in the old E[X²] formulation).
            global_var = global_sq - global_mean.pow(2)

            if not dry_run:
                alpha = self.ema_zscore_alpha
                # During the first `ema_zscore_trend_warmup` steps, weights move
                # non-linearly (LR warmup regime) — fitting Holt's trend here makes
                # it memorize noise and forecast phantom drift. Run plain EMA +
                # bias correction until then; trend buffer stays zero, so the
                # forecast `ℓ + b = ℓ` matches plain-EMA output at the transition
                # step (seamless switch). Trend applies to the MEAN only; variance
                # is always plain EMA.
                use_trend = (
                    self.ema_zscore_trend
                    and self._ema_step_py >= self.ema_zscore_trend_warmup
                )
                if use_trend:
                    assert self._ema_mean_trend is not None
                    beta = self.ema_zscore_trend_beta
                    mean_trend = cast(torch.Tensor, self._ema_mean_trend)
                    self._holt_update(ema_mean, mean_trend, global_mean, alpha, beta)
                else:
                    ema_mean.lerp_(global_mean, 1.0 - alpha)
                ema_var.lerp_(global_var, 1.0 - alpha)
                # Keep the checkpointed buffer and the Python shadow in lockstep — both
                # advance together so `_apply_ema_zscore` can read the Python int without
                # a device→host sync on the forward hot path.
                ema_step.add_(1)
                self._ema_step_py += 1

            # Reset accumulators whether or not we applied the update — never leak.
            self._ema_logit_sum_accum = None
            self._ema_logit_sq_sum_accum = None
            self._ema_token_count_accum = 0

    @staticmethod
    def _holt_update(
        level: torch.Tensor,
        trend: torch.Tensor,
        observation: torch.Tensor,
        alpha: float,
        beta: float,
    ) -> None:
        """
        One step of textbook undamped Holt's linear-trend smoothing, in-place on
        ``level`` and ``trend``.

        Meaning of each argument in *this* router's context — Holt's is applied to
        the per-expert mean logit, so all three tensors are shape ``(num_experts,)``:

        - ``level`` — running estimate of the per-expert mean logit *right now*
          (Holt's symbol ``ℓ``). Equivalent to ``_ema_mean``.
        - ``trend`` — running estimate of how much the per-expert mean logit is
          drifting per optimizer step (Holt's ``b``). Equivalent to
          ``_ema_mean_trend``. Zero when the mean is stationary; positive when the
          logits are drifting up, negative when drifting down.
        - ``observation`` — the current batch's per-expert sample mean (Holt's
          ``y_t``), computed as ``sum / count`` over this step's tokens after the
          distributed all-reduce.
        - ``alpha`` — decay for ``level`` (old-weight convention, ≈1 = slow). At
          α=0.99, ``level`` has ~100-step memory of its observations.
        - ``beta`` — decay for ``trend`` (same convention). At β=0.9, ``trend``
          has ~10-step memory of level-differences.

        Update equations (Hyndman-style old-weight convention, α, β ∈ (0,1))::

            ℓ_t = α·(ℓ_{t-1} + b_{t-1}) + (1-α)·y_t
            b_t = β·b_{t-1}            + (1-β)·(ℓ_t - ℓ_{t-1})

        In words: the new level blends the previous level+trend forecast with the
        current observation; the new trend blends the previous trend with the
        actual change in level we just observed.

        The 1-step-ahead forecast used at forward-time is ``ℓ_t + b_t`` — this has
        zero steady-state lag for linearly-drifting input, which is why we can't
        replace it with plain EMA for the mean (plain EMA has an ``α/(1-α) ≈ 99``
        step lag at α=0.99, which compounds into routing imbalance — see
        EMA_ZSCORE_ANALYSIS.md). Used for the mean only; variance uses plain EMA.
        """
        old_level = level.clone()
        level.mul_(alpha).add_(trend, alpha=alpha).add_(observation, alpha=1.0 - alpha)
        trend.mul_(beta).add_(level - old_level, alpha=1.0 - beta)

    def _bias_corrected_ema_stats(self, step: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return the per-expert forecasted mean and std that :meth:`_apply_ema_zscore`
        uses to normalize router logits.

        Adam-style bias correction ``1 / (1 - α^t)`` is applied on every path. For
        the Holt path this also corrects the zero-init bias of the level (``ℓ`` is
        scaled by ``1 - α^t`` early on) and has negligible effect once
        ``t ≫ 1/(1-α)``, where it fades to 1. Applying the correction here makes
        the transition at ``step == warmup`` exactly continuous, since the trend
        starts at 0 so ``(ℓ + 0)/(1 - α^t) = ℓ/(1 - α^t)`` matches the plain-EMA
        forecast.

        Mean path:
            plain EMA (step < warmup or trend disabled):  ``μ̂ = ℓ_μ / (1 - α^t)``
            Holt's    (step ≥ warmup):                    ``μ̂ = (ℓ_μ + b_μ) / (1 - α^t)``

        Variance path (always plain EMA):                 ``σ̂² = ℓ_var / (1 - α^t)``

        Caller must ensure ``step > 0`` (for ``t=0`` the EMA is uninitialized).
        """
        assert self._ema_mean is not None and self._ema_var is not None
        ema_mean = cast(torch.Tensor, self._ema_mean)
        ema_var = cast(torch.Tensor, self._ema_var)
        bias_correction = 1.0 - (self.ema_zscore_alpha**step)
        if self.ema_zscore_trend and step >= self.ema_zscore_trend_warmup:
            assert self._ema_mean_trend is not None
            mean_trend = cast(torch.Tensor, self._ema_mean_trend)
            mean_hat = (ema_mean + mean_trend) / bias_correction
        else:
            mean_hat = ema_mean / bias_correction
        # Std floor of 1e-2 (clamp on var = 1e-4) is a belt-and-suspenders safety:
        # under the direct-Var formulation it cannot be driven below the minimum
        # observed Var (EMA is a convex combination of non-negative observations),
        # so the floor should only trigger on genuine variance collapse — e.g.
        # immediately post-checkpoint-load before stats refill. If you see it
        # trigger during training, something upstream is wrong.
        ema_std = (ema_var / bias_correction).clamp(min=1e-4).sqrt()
        return mean_hat, ema_std

    def _apply_ema_zscore(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Z-normalize ``logits`` per expert using the EMA snapshot from the previous
        optimizer step, then accumulate raw stats (sum, sum-of-squares, token count)
        so that :meth:`post_batch` can compute the sample-based variance observation
        ``var_obs = sq_sum/count − (sum/count)²`` and apply a single EMA update per
        step. Gradient flows through ``logits`` only — EMA buffers and accumulators
        are detached.

        The EMA snapshot is constant across all microbatches within a step, so all
        microbatches see identical ``μ̂``, ``σ̂``. This makes the per-step EMA decay
        unambiguous (one ``α``-update per optimizer step, not per microbatch).

        Adam-style bias correction (``/ (1 - α^t)``) is applied so that the EMA
        gives unbiased mean / variance estimates from step 1 onwards, instead of
        taking ~``1/(1-α)`` steps to ramp up from the zero init.

        Step ``t = 0`` (no stats yet) returns ``logits`` unchanged — the very first
        forward sees raw logits, which is acceptable for one batch and avoids
        normalizing by an undefined std.
        """
        # Read the Python shadow, not `_ema_step_count.item()` — the buffer exists for
        # checkpointing, the int for avoiding a per-microbatch device→host sync.
        step = self._ema_step_py
        num_experts = logits.shape[-1]

        if step == 0:
            normalized = logits
        else:
            mean_hat, ema_std = self._bias_corrected_ema_stats(step)
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

        # Per-expert token share (sums to 1.0). In W&B, plot all on one chart with
        # metric regex ``train/block .*/expert_.*/tokens percentage``.
        expert_fraction = batch_size_per_expert.float() / batch_size_per_expert.sum()
        for i in range(expert_fraction.shape[0]):
            out[f"expert {i:02d}/tokens percentage"] = (expert_fraction[i], ReduceType.mean)

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

        # Log the bias-corrected per-expert mean/std that `_apply_ema_zscore` actually
        # applies, not the raw (biased) accumulators.
        if self.ema_zscore_normalize:
            assert self._ema_mean is not None and self._ema_step_count is not None
            ema_step = cast(torch.Tensor, self._ema_step_count)
            step = self._ema_step_py
            if step > 0:
                mean_hat, ema_std = self._bias_corrected_ema_stats(step)
            else:
                mean_hat = cast(torch.Tensor, self._ema_mean)
                ema_std = torch.zeros_like(mean_hat)
            for i in range(mean_hat.shape[0]):
                out[f"expert {i:02d}/ema mean"] = (mean_hat[i], ReduceType.mean)
                out[f"expert {i:02d}/ema std"] = (ema_std[i], ReduceType.mean)
            out["ema step"] = (ema_step.float(), ReduceType.mean)

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

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        # `.grad` is populated between backward and optim.step — see
        # `TransformerTrainModule.train_batch`. Useful for detecting router weight runaway
        # under ema-zscore feedback loops.
        out = super().compute_metrics(reset=False)
        if self.weight.grad is not None:
            full_grad = get_full_tensor(self.weight.grad.detach()).float()
            per_expert_grad_norm = full_grad.view(self.num_experts, self.d_model).norm(dim=1)
            for i in range(per_expert_grad_norm.shape[0]):
                out[f"expert {i:02d}/weight grad norm"] = (
                    per_expert_grad_norm[i],
                    ReduceType.mean,
                )
        if reset:
            self.reset_metrics()
        return out

    def apply_tp(self, tp_mesh: DeviceMesh, float8_enabled: bool = False):
        super().apply_tp(tp_mesh, float8_enabled=float8_enabled)
        self.register_parameter(
            "weight", nn.Parameter(distribute_tensor(self.weight, tp_mesh, [Replicate()]))
        )
