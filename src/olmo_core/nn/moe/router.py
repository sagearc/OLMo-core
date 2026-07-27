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
    "MoEBinaryLeaveOneOutLinearRouter",
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


_uniform_expert_assignment: Callable[[torch.Tensor, int], torch.Tensor] = (
    _UniformExpertAssignment.apply
)  # type: ignore


class MoERouterType(StrEnum):
    """
    An enumeration of the different MoE router implementations.
    """

    default = "default"
    """
    ➡️ :class:`MoELinearRouter`
    """

    binary_leave_one_out = "binary_leave_one_out"
    """
    ➡️ :class:`MoEBinaryLeaveOneOutLinearRouter`
    """

    half_leave_one_out = "half_leave_one_out"
    """
    ➡️ :class:`MoEHalfLeaveOneOutLinearRouter`
    """

    centroid = "centroid"
    """
    ➡️ :class:`MoECentroidRouter`
    """


class MoERouterGatingFunction(StrEnum):
    softmax = "softmax"
    sigmoid = "sigmoid"
    identity = "identity"
    """
    Pass scores through unchanged. Used by :class:`MoECentroidRouter` where cosine
    similarity is already in ``[-1, 1]`` and needs no further nonlinearity.
    """


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
    centroid_alpha: float = 0.99
    """
    Static EMA decay for :class:`MoECentroidRouter` centroid tracking. Used when
    ``name == MoERouterType.centroid`` and ``centroid_lr_lambda`` is ``None``.
    """
    centroid_lr_lambda: Optional[float] = None
    """
    When set, replaces the static ``centroid_alpha`` with a learning-rate-coupled rate:

    .. math::

        1 - \\alpha_t = \\lambda_{\\text{EMA}} \\cdot \\eta_t

    so the centroid step size tracks the optimizer's current LR and decays
    naturally with the cosine schedule.  ``centroid_alpha`` is ignored when
    this is set.  Only used when ``name == MoERouterType.centroid``.
    """
    bias_lr_lambda: Optional[float] = None
    """
    When set, replaces the static ``bias_gamma`` with a learning-rate-coupled rate:

    .. math::

        \\gamma_t = \\lambda_{\\text{bias}} \\cdot \\eta_t

    so the dual ascent step size also decays with the LR schedule.
    ``bias_gamma`` is ignored when this is set.
    """
    centroid_spherical: bool = False
    """
    When ``True``, the centroid EMA update normalizes the observed cluster mean to a
    unit vector before the lerp step, making the primal M-step consistent with the
    cosine similarity routing criterion (spherical k-means M-step).  Default ``False``
    preserves the standard k-means M-step (raw hidden-state mean), which can cause
    centroid norms to drift away from the unit sphere.  Only used when
    ``name == MoERouterType.centroid``.
    """
    num_centroids_per_expert: int = 1
    """
    Number of sub-centroids per expert.  With ``C > 1``, each expert maintains ``C``
    centroid vectors.  The routing score for expert ``k`` is the maximum cosine
    similarity across its ``C`` sub-centroids, and the M-step updates only the
    winning sub-centroid (winner-takes-all within each expert).  This allows a single
    expert to cover multiple modes in its token distribution.  Only used when
    ``name == MoERouterType.centroid``.
    """
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
    ema_zscore_trend_damping: float = 1.0
    """
    Gardner-McKenzie damping φ on the trend (Gardner 1985). Forecast becomes
    ``μ̂ = ℓ + φ·b`` and the level update becomes
    ``ℓ_t = α·(ℓ_{t-1} + φ·b_{t-1}) + (1-α)·y_t``. At φ=1.0 this reduces to
    undamped Holt's (current default, zero steady-state lag). At φ<1, the forecast
    read at step ``t`` lags the true mean ``y_t`` by ``(1-φ)·m/(1-α)`` under drift
    rate ``m``, which creates a z-score offset ``(1-φ)·m/((1-α)·σ̂)`` that
    systematically suppresses the softmax share for drifting experts — i.e. an
    implicit gradient-suppression regularizer on router weight magnitude.
    Tradeoff:

    - φ=1.0 (default): no lag, zero routing centering bias, no implicit
      regularization. μ̂ magnitude is bounded only by weight decay.
    - φ≈0.9: ~30% gradient suppression on high-drift experts (`m/σ̂ ≈ 0.04`),
      routing centering bias ~0.4σ, meaningful reduction in μ̂ equilibrium.
    - φ→0: approaches plain EMA's logarithmic-saturation regime (Lambert-W);
      strong regularization, large centering bias.

    See ``EMA_ZSCORE_ANALYSIS.md`` §4 for the tradeoff argument; this knob lets
    you explicitly choose a middle point. Only used when ``ema_zscore_trend`` is
    enabled. Must be in (0, 1].
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
        if self.name in (
            MoERouterType.default,
            MoERouterType.binary_leave_one_out,
            MoERouterType.half_leave_one_out,
        ):
            num_params += d_model * num_experts
        elif self.name != MoERouterType.centroid:
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
        try:
            if self.name in (
                MoERouterType.default,
                MoERouterType.binary_leave_one_out,
                MoERouterType.half_leave_one_out,
            ):
                kwargs.pop("centroid_alpha", None)
                kwargs.pop("centroid_lr_lambda", None)
                kwargs.pop("centroid_spherical", None)
                kwargs.pop("num_centroids_per_expert", None)
                if self.dtype is not None:
                    kwargs["dtype"] = self.dtype.as_pt()
                elif dtype is not None:
                    kwargs["dtype"] = dtype
                if self.name == MoERouterType.binary_leave_one_out:
                    return MoEBinaryLeaveOneOutLinearRouter(**kwargs)
                if self.name == MoERouterType.half_leave_one_out:
                    return MoEHalfLeaveOneOutLinearRouter(**kwargs)
                return MoELinearRouter(**kwargs)
            elif self.name == MoERouterType.centroid:
                return MoECentroidRouter(**kwargs)
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
        bias_lr_lambda: Optional[float] = None,
        gating_function: MoERouterGatingFunction = MoERouterGatingFunction.softmax,
        seq_aux_loss_weight: Optional[float] = None,
        ema_zscore_normalize: bool = False,
        ema_zscore_alpha: float = 0.99,
        ema_zscore_trend: bool = False,
        ema_zscore_trend_beta: float = 0.9,
        ema_zscore_trend_warmup: int = 1000,
        ema_zscore_trend_damping: float = 1.0,
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
        self.bias_lr_lambda = bias_lr_lambda
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
            assert 0.0 < ema_zscore_trend_beta < 1.0, (
                f"ema_zscore_trend_beta must be in (0, 1), got {ema_zscore_trend_beta}."
            )
            assert ema_zscore_trend_warmup >= 0, (
                f"ema_zscore_trend_warmup must be >= 0, got {ema_zscore_trend_warmup}."
            )
            assert 0.0 < ema_zscore_trend_damping <= 1.0, (
                f"ema_zscore_trend_damping must be in (0, 1], got {ema_zscore_trend_damping}."
            )
        self.ema_zscore_trend_beta = ema_zscore_trend_beta
        self.ema_zscore_trend_warmup = ema_zscore_trend_warmup
        self.ema_zscore_trend_damping = ema_zscore_trend_damping
        self.lb_loss_weight = lb_loss_weight
        self.lb_loss_granularity = lb_loss_granularity
        self.z_loss_weight = z_loss_weight
        self.group: Optional[dist.ProcessGroup] = None
        self.cp_mesh: Optional[dist.DeviceMesh] = None
        self.tp_mesh: Optional[dist.DeviceMesh] = None

        if self._bias_enabled:
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

    @property
    def _bias_enabled(self) -> bool:
        return self.bias_gamma is not None or self.bias_lr_lambda is not None

    def reset_parameters(self):
        self._batch_size_per_expert = hide_from_torch(
            torch.zeros(self.num_experts, device=self.device)
        )

        if self._bias_enabled:
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
        if self._bias_enabled:
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
    def post_batch(self, dry_run: bool = False, lr: Optional[float] = None):
        if not self.training:
            return

        # ---- DeepSeek bias-rule update --------------------------------------------------
        if self.bias_lr_lambda is not None and lr is not None:
            effective_gamma = self.bias_lr_lambda * lr
        else:
            effective_gamma = self.bias_gamma

        if effective_gamma is not None:
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
                effective_gamma * (ideal_batch_size_per_expert - batch_size_per_expert).sign()
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
                    self.ema_zscore_trend and self._ema_step_py >= self.ema_zscore_trend_warmup
                )
                if use_trend:
                    assert self._ema_mean_trend is not None
                    beta = self.ema_zscore_trend_beta
                    phi = self.ema_zscore_trend_damping
                    mean_trend = cast(torch.Tensor, self._ema_mean_trend)
                    self._holt_update(ema_mean, mean_trend, global_mean, alpha, beta, phi)
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
        phi: float = 1.0,
    ) -> None:
        """
        One step of Holt's linear-trend smoothing with optional Gardner-McKenzie
        damping, in-place on ``level`` and ``trend``.

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
        - ``phi`` — Gardner-McKenzie damping (Gardner 1985). φ=1.0 is undamped
          Holt's (default). φ<1 damps the trend contribution inside the level
          update, creating an intentional forecast lag that acts as an implicit
          gradient-suppression regularizer on drifting experts.

        Update equations (Hyndman-style old-weight convention, α, β ∈ (0,1),
        φ ∈ (0,1])::

            ℓ_t = α·(ℓ_{t-1} + φ·b_{t-1}) + (1-α)·y_t
            b_t = β·b_{t-1}                + (1-β)·(ℓ_t - ℓ_{t-1})

        In words: the new level blends the previous level+damped-trend forecast
        with the current observation; the new trend blends the previous trend with
        the actual change in level we just observed. With φ=1 this is the textbook
        undamped update.

        The 1-step-ahead forecast used at forward-time is ``ℓ_t + φ·b_t``. At φ=1
        this has zero steady-state lag for linear drift (the undamped behavior
        that motivated switching away from plain EMA). At φ<1 the forecast read
        at step ``t`` lags the current mean ``y_t`` by ``(1-φ)·m/(1-α)`` under
        drift rate ``m`` — see :meth:`_bias_corrected_ema_stats` for how this is
        applied.
        """
        old_level = level.clone()
        level.mul_(alpha).add_(trend, alpha=alpha * phi).add_(observation, alpha=1.0 - alpha)
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
            Holt's    (step ≥ warmup):                    ``μ̂ = (ℓ_μ + φ·b_μ) / (1 - α^t)``
              where φ is ``ema_zscore_trend_damping`` (default 1.0 = undamped).

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
            phi = self.ema_zscore_trend_damping
            mean_hat = (ema_mean + phi * mean_trend) / bias_correction
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
        if not self._bias_enabled:
            if self.top_k == 1:
                expert_weights, expert_indices = scores.max(dim=-1, keepdim=True)
            else:
                expert_weights, expert_indices = torch.topk(scores, self.top_k, dim=-1)
        else:
            assert self.score_bias is not None
            with torch.no_grad():
                _, expert_indices = torch.topk(
                    scores + self.score_bias.unsqueeze(0),
                    self.top_k,
                    dim=-1,  # type: ignore
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
        elif self.gating_function == MoERouterGatingFunction.identity:
            scores = gating_logits
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
            if self._bias_enabled:
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

        def geometry_metrics(
            rows: torch.Tensor, prefix: str
        ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
            rows = rows.view(self.num_experts, self.d_model).float()
            mean_row = rows.mean(dim=0, keepdim=True)
            centered = rows - mean_row
            total_energy = rows.square().sum()
            centered_energy = centered.square().sum()
            tiny = torch.finfo(rows.dtype).tiny
            common_energy_fraction = (
                self.num_experts * mean_row.square().sum()
            ) / total_energy.clamp_min(tiny)

            singular_energy = torch.linalg.svdvals(centered).square()
            singular_probability = singular_energy / singular_energy.sum().clamp_min(tiny)
            effective_rank = torch.where(
                centered_energy > 0,
                torch.exp(
                    -(singular_probability * singular_probability.clamp_min(tiny).log()).sum()
                ),
                torch.zeros_like(centered_energy),
            )

            eye = torch.eye(self.num_experts, dtype=torch.bool, device=rows.device)
            raw_cosine = F.normalize(rows, dim=-1) @ F.normalize(rows, dim=-1).T
            centered_cosine = F.normalize(centered, dim=-1) @ F.normalize(centered, dim=-1).T
            return {
                f"{prefix} common energy fraction": (
                    common_energy_fraction,
                    ReduceType.mean,
                ),
                f"{prefix} centered effective rank": (
                    effective_rank,
                    ReduceType.mean,
                ),
                f"{prefix} raw abs cosine": (
                    raw_cosine.masked_select(~eye).abs().mean(),
                    ReduceType.mean,
                ),
                f"{prefix} centered abs cosine": (
                    centered_cosine.masked_select(~eye).abs().mean(),
                    ReduceType.mean,
                ),
            }

        # `.grad` is populated between backward and optim.step — see
        # `TransformerTrainModule.train_batch`. Useful for detecting router weight runaway
        # under ema-zscore feedback loops.
        out = super().compute_metrics(reset=False)
        full_weight = get_full_tensor(self.weight.detach()).float()
        out.update(geometry_metrics(full_weight, "weight geometry"))
        per_expert_weight_norm = full_weight.view(self.num_experts, self.d_model).norm(dim=1)
        for i in range(per_expert_weight_norm.shape[0]):
            out[f"expert {i:02d}/weight norm"] = (
                per_expert_weight_norm[i],
                ReduceType.mean,
            )
        if self.weight.grad is not None:
            full_grad = get_full_tensor(self.weight.grad.detach()).float()
            out.update(geometry_metrics(full_grad, "gradient geometry"))
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


class MoEBinaryLeaveOneOutLinearRouter(MoELinearRouter):
    """
    Two-expert top-1 router with forward scores
    ``(1 - sigmoid(z_1), 1 - sigmoid(z_0))``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_experts != 2 or self.top_k != 1:
            raise OLMoConfigurationError(
                "binary leave-one-out routing requires 2 experts and top_k=1."
            )
        if self.gating_function != MoERouterGatingFunction.sigmoid:
            raise OLMoConfigurationError(
                "binary leave-one-out routing requires sigmoid scores."
            )
        if self.normalize_expert_weights is not None:
            raise OLMoConfigurationError(
                "binary leave-one-out routing requires no top-1 weight normalization."
            )

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # `scores` includes the ordinary router's 1e-7 numerical floor.
        leave_one_out_scores = 1.0 - (scores - 1e-7).flip(dims=(-1,)) + 1e-7
        return MoERouter.get_top_k(self, leave_one_out_scores)


class MoEHalfLeaveOneOutLinearRouter(MoELinearRouter):
    """
    Forward-matched half-selected / complementary-gradient router.

    This router requires exactly half of the experts to be selected. Its forward
    pass is identical to the ordinary router. For the backward pass, the selected
    scores are rank-paired with the complementary half: best selected with worst
    unselected, second-best selected with second-worst unselected, and so on.
    Each selected forward path differentiates through ``1 - s`` for its paired
    unselected expert.

    The no-selected-row-gradient invariant requires independent per-expert gates.
    This implementation consequently supports sigmoid gating only; a full
    softmax would couple every score through its denominator.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.num_experts % 2 != 0 or self.top_k != self.num_experts // 2:
            raise OLMoConfigurationError(
                "half leave-one-out routing requires an even number of experts "
                "and top_k == num_experts / 2."
            )
        if self.gating_function != MoERouterGatingFunction.sigmoid:
            raise OLMoConfigurationError(
                "half leave-one-out routing requires independent sigmoid scores; "
                "full softmax scores would leak gradient into selected rows."
            )

    def get_top_k(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        expert_weights, expert_indices = super().get_top_k(scores)

        selected = torch.zeros_like(scores, dtype=torch.bool)
        selected.scatter_(-1, expert_indices, True)

        if self._bias_enabled:
            assert self.score_bias is not None
            ranking_scores = scores + self.score_bias.unsqueeze(0)
        else:
            ranking_scores = scores
        with torch.no_grad():
            _, complement_indices = torch.topk(
                ranking_scores.masked_fill(selected, torch.inf),
                self.top_k,
                dim=-1,
                largest=False,
            )
        backward_weights = 1.0 - scores.gather(-1, complement_indices)

        # Forward values and assignments exactly match the ordinary router.
        # Only the autograd dependency is replaced.
        return (
            expert_weights.detach() + (backward_weights - backward_weights.detach()),
            expert_indices,
        )


class MoECentroidRouter(MoERouter):
    """
    Gradient-free EMA centroid router implementing the primal-dual algorithm for
    capacity-constrained online clustering (see paper §3).

    Each expert ``k`` maintains a centroid ``c_k ∈ R^{d_model}`` that tracks the
    running mean of hidden states routed to it.  Routing logits are the raw dot
    product ``h · c_k``; the gating function (softmax / sigmoid / identity) and
    ``normalize_expert_weights`` are orthogonal config choices that determine how
    those logits become per-token expert weights.  No gradient ever flows into
    ``c_k`` — it is a pure statistic.

    **Primal update** (centroid tracking, M-step):

    .. code-block::

        c_k  ←  α·c_k + (1-α)·mean_{assigned}(h(x))   [only for experts with tokens]

    **Dual update** (load-balancing, via ``bias_gamma``):

    .. code-block::

        b_k  ←  b_k + γ·sign(τ - f_k)

    where ``τ = 1/K`` is the target token fraction and ``f_k`` is the measured
    fraction.  This is the subgradient step on the Lagrange multiplier for the
    uniform-coverage constraint in the Lagrangian (see paper §3.2).

    Together they implement alternating primal-dual optimisation on:

    .. math::

        \\max_{Z,C} \\mathbb{E}_x\\bigl[\\sum_k z_k(x)\\,\\langle h(x), c_k\\rangle\\bigr]
        \\quad \\text{s.t.} \\quad \\mathbb{E}_x[z_k(x)] = \\tau \\;\\forall k

    without any auxiliary loss term and without any inter-centroid gradient
    coupling.
    """

    def __init__(
        self,
        *,
        centroid_alpha: float = 0.99,
        centroid_lr_lambda: Optional[float] = None,
        centroid_spherical: bool = False,
        num_centroids_per_expert: int = 1,
        init_device: str = "cpu",
        **kwargs,
    ):
        assert 0.0 < centroid_alpha < 1.0, (
            f"centroid_alpha must be in (0, 1), got {centroid_alpha}."
        )
        assert num_centroids_per_expert >= 1, (
            f"num_centroids_per_expert must be >= 1, got {num_centroids_per_expert}."
        )
        super().__init__(init_device=init_device, **kwargs)
        self.centroid_alpha = centroid_alpha
        self.centroid_lr_lambda = centroid_lr_lambda
        self.centroid_spherical = centroid_spherical
        self.num_centroids_per_expert = num_centroids_per_expert

        # Random unit vectors — shape (K*C, d_model).
        centroid_init = torch.randn(
            self.num_experts * num_centroids_per_expert,
            self.d_model,
            dtype=torch.float32,
            device=init_device,
        )
        F.normalize(centroid_init, dim=-1, out=centroid_init)
        self.register_buffer("_centroid", centroid_init)
        self.register_buffer(
            "_centroid_step",
            torch.zeros((), dtype=torch.long, device=init_device),
        )
        # Accumulators reset each optimizer step.  Hidden from torch so FSDP
        # and torch.compile don't manage them.
        self._centroid_sum_accum: Optional[_HiddenTensor] = None
        self._centroid_count_accum: Optional[_HiddenTensor] = None
        # Python shadow of _centroid_step to avoid device→host syncs on the
        # forward hot path.
        self._centroid_step_py: int = 0

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self._centroid_step is not None:
            self._centroid_step_py = int(cast(torch.Tensor, self._centroid_step).item())
        # Discard any in-flight microbatch accumulations — they belong to the
        # pre-checkpoint forward pass and are invalid after a state-dict load.
        self._centroid_sum_accum = None
        self._centroid_count_accum = None

    def reset_parameters(self):
        super().reset_parameters()
        if self._centroid is not None:
            centroid = cast(torch.Tensor, self._centroid)
            nn.init.normal_(centroid)
            F.normalize(centroid, dim=-1, out=centroid)
        if self._centroid_step is not None:
            cast(torch.Tensor, self._centroid_step).zero_()
        self._centroid_step_py = 0
        self._centroid_sum_accum = None
        self._centroid_count_accum = None

    @property
    def device(self) -> torch.device:
        centroid = cast(torch.Tensor, self._centroid)
        return centroid.device if centroid.device.type != "meta" else torch.device("cpu")

    def get_expert_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Raw dot product between each token and each centroid.

        Centroids are random unit vectors from init, so all experts are immediately
        valid. Score magnitudes scale with ``‖h‖ · ‖c_k‖``, so depending on the
        chosen ``gating_function`` (softmax over all experts, sigmoid, or identity)
        and on whether ``normalize_expert_weights`` is set, downstream gating may
        saturate or peak sharply — these are orthogonal config choices the caller
        controls.

        :returns: Dot products of shape ``(*, num_experts)``.
        """
        centroid = cast(torch.Tensor, self._centroid)
        flat = x.float().view(-1, self.d_model)
        score = flat @ centroid.t()  # (N, K*C)
        if self.num_centroids_per_expert > 1:
            # Score for expert k = max raw dot over its C sub-centroids.
            score = score.view(-1, self.num_experts, self.num_centroids_per_expert).amax(dim=-1)
        return score.view(*x.shape[:-1], self.num_experts)

    @torch.no_grad()
    def _accumulate_centroid(self, flat_h: torch.Tensor, expert_indices: torch.Tensor) -> None:
        """
        :param flat_h: Hidden states, shape ``(N, d_model)``, float32.
                       Unit-norm when ``centroid_spherical=True``, raw otherwise.
        :param expert_indices: Expert assignment indices, shape ``(N, top_k)``, in ``[0, K)``.
        """
        N, d = flat_h.shape
        K = self.num_experts
        C = self.num_centroids_per_expert
        # Repeat each token's hidden state for each of its top_k assignments.
        h_rep = flat_h.unsqueeze(1).expand(-1, self.top_k, -1).reshape(N * self.top_k, d)
        expert_idx = expert_indices.reshape(-1).long()  # (N*top_k,) in [0, K)

        if C == 1:
            global_idx = expert_idx
        else:
            # Winner-takes-all within each expert: find the sub-centroid with highest
            # cosine similarity to the token and accumulate only to that one.
            centroid = cast(torch.Tensor, self._centroid)  # (K*C, d)
            assigned_c = centroid.reshape(K, C, d)[expert_idx]  # (N*top_k, C, d)
            # h_rep is already unit-norm when centroid_spherical=True; normalize otherwise.
            h_unit = (h_rep if self.centroid_spherical else F.normalize(h_rep, dim=-1)).unsqueeze(1)
            F.normalize(assigned_c, dim=-1, out=assigned_c)
            sims = (h_unit * assigned_c).sum(-1)  # (N*top_k, C)
            winning_sub = sims.argmax(dim=-1)  # (N*top_k,) in [0, C)
            global_idx = expert_idx * C + winning_sub  # (N*top_k,) in [0, K*C)

        centroid_sum = torch.zeros(K * C, d, dtype=torch.float32, device=flat_h.device)
        centroid_count = torch.zeros(K * C, dtype=torch.float32, device=flat_h.device)
        centroid_sum.index_add_(0, global_idx, h_rep)
        centroid_count.index_add_(0, global_idx, torch.ones_like(global_idx, dtype=torch.float32))

        if self._centroid_sum_accum is None:
            self._centroid_sum_accum = hide_from_torch(centroid_sum)
            self._centroid_count_accum = hide_from_torch(centroid_count)
        else:
            unhide_from_torch(self._centroid_sum_accum).add_(centroid_sum)
            unhide_from_torch(self._centroid_count_accum).add_(centroid_count)

    @torch.no_grad()
    def post_batch(self, dry_run: bool = False, lr: Optional[float] = None) -> None:
        # Dual update (score_bias rule).
        super().post_batch(dry_run=dry_run, lr=lr)

        # Primal update: EMA centroid step.
        if self._centroid_sum_accum is None:
            return

        if self.centroid_lr_lambda is not None and lr is not None:
            one_minus_alpha = self.centroid_lr_lambda * lr
        else:
            one_minus_alpha = 1.0 - self.centroid_alpha
        if not (0.0 < one_minus_alpha <= 1.0):
            raise OLMoConfigurationError(
                f"centroid step size must be in (0, 1]; got {one_minus_alpha:.6f} "
                f"(centroid_lr_lambda={self.centroid_lr_lambda}, lr={lr})"
            )

        centroid = cast(torch.Tensor, self._centroid)
        centroid_step = cast(torch.Tensor, self._centroid_step)
        sum_accum = unhide_from_torch(self._centroid_sum_accum)
        count_accum = unhide_from_torch(self._centroid_count_accum)

        if is_distributed():
            dist.all_reduce(sum_accum, group=self.group)
            dist.all_reduce(count_accum, group=self.group)

        has_tokens = count_accum > 0
        if not dry_run:
            if has_tokens.any():
                obs_mean = sum_accum[has_tokens] / count_accum[has_tokens].unsqueeze(-1)
                if self.centroid_spherical:
                    obs_mean = F.normalize(obs_mean, dim=-1)
                # Boolean indexing returns a copy; lerp_ on it would be a no-op.
                # Use out-of-place lerp and assign back to update the buffer in-place.
                # Only active experts update; cold centroids stay put (no decay toward zero).
                updated = centroid[has_tokens].lerp(obs_mean, one_minus_alpha)
                if self.centroid_spherical:
                    # Renormalize to keep buffer on the unit sphere (canonical spherical k-means).
                    # get_expert_logits normalizes at query time anyway, but keeping the buffer
                    # unit-norm makes centroid-norm metrics interpretable.
                    updated = F.normalize(updated, dim=-1)
                centroid[has_tokens] = updated
            centroid_step.add_(1)
            self._centroid_step_py += 1

        self._centroid_sum_accum = None
        self._centroid_count_accum = None

    def forward(
        self,
        x: torch.Tensor,
        *,
        loss_div_factor: Optional[Union[torch.Tensor, float]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        expert_weights, expert_indices, batch_size_per_expert, aux_loss = super().forward(
            x, loss_div_factor=loss_div_factor
        )
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                flat_h = x.view(-1, self.d_model).float()
                if self.centroid_spherical:
                    flat_h = F.normalize(flat_h, dim=-1)
                self._accumulate_centroid(flat_h, expert_indices.view(-1, self.top_k))
        return expert_weights, expert_indices, batch_size_per_expert, aux_loss

    def reset_metrics(self):
        super().reset_metrics()

    @torch.no_grad()
    def compute_metrics(
        self, reset: bool = True
    ) -> Dict[str, Tuple[torch.Tensor, Optional["ReduceType"]]]:
        from olmo_core.train.common import ReduceType

        out = super().compute_metrics(reset=False)
        step = self._centroid_step_py
        if step > 0:
            centroid = cast(torch.Tensor, self._centroid)
            if self.centroid_lr_lambda is None:
                # Static alpha: bias correction 1/(1-α^t) is exact.
                bc = 1.0 - self.centroid_alpha**step
                centroid_hat = centroid / bc
            else:
                # Variable alpha (LR-coupled): per-step alphas differ, so 1/(1-α^t) is
                # invalid. Log the raw buffer norm — still useful for diagnosing collapse.
                centroid_hat = centroid
            norms = centroid_hat.norm(dim=-1)
            C = self.num_centroids_per_expert
            for i in range(norms.shape[0]):
                key = (
                    f"expert {i // C:02d}/sub {i % C}/centroid norm"
                    if C > 1
                    else f"expert {i:02d}/centroid norm"
                )
                out[key] = (norms[i], ReduceType.mean)
        out["centroid step"] = (
            cast(torch.Tensor, self._centroid_step).float(),
            ReduceType.mean,
        )
        if reset:
            self.reset_metrics()
        return out
