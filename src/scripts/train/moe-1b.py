"""
1B MoE routing-mechanism ablation, paper-exact replication of the DeepSeek
auxiliary-loss-free 1B ablation architecture (arXiv:2408.15664 §4 Table 5).

ALL variants share an identical architecture / optimizer / batch / schedule;
only the routing strategy and its associated auxiliary losses differ. This is
the single source of truth — adding a new routing variant means adding ONE
branch to ``configure_routing()``.

Architecture (Table 5, "1B" column):
  d_model=1024, n_layers=9, n_heads=8
  num_experts=64 routed, top_k=6
  expert_hidden=512, shared_expert_hidden=1024 (= N_s=2 at expert size)

Optimizer / schedule (Table 5):
  AdamW(β=(0.9, 0.95), wd=0.1), grad-clip=1.0
  peak lr=1e-3, cosine to 1e-4 (alpha_f=0.1), warmup=1000 steps
  global batch = 1152 sequences × 2048 tokens = 2.36M tokens/step
  max_duration = 100B tokens (~42 400 steps)

Usage:
    python src/scripts/train/moe-1b.py RUN_NAME --routing=VARIANT [OVERRIDES...]

    where VARIANT is one of:
        baseline          softmax + Switch lb_loss (0.01) + router z-loss (0.001) — floor
        deepseek          sigmoid + bias rule (γ=1e-3) — strict arXiv:2408.15664, no aux loss
        ema               EMA z-norm + softmax (proposed) — no aux loss, no z-loss
        ema_trend         as `ema`, plus undamped Holt's linear-trend smoothing on MEAN —
                          zero steady-state lag for linear μ drift; variance stays plain EMA
        ema_trend_damped  as `ema_trend`, with Gardner-McKenzie damping φ=0.9 on the trend —
                          implicit gradient-suppression regularizer on μ̂ magnitude at cost
                          of ~0.4σ routing centering bias
        ema_centroid      gradient-free EMA centroid routing + dual bias — primal-dual
                          algorithm for capacity-constrained clustering; no router params,
                          no aux losses, zero inter-centroid gradient coupling
        ema_centroid_sph  as `ema_centroid` but with spherical k-means M-step: observed
                          cluster mean is L2-normalised before the lerp, keeping centroids
                          on the unit sphere — consistent with the cosine similarity score
        ema_centroid_sph_c2  as `ema_centroid_sph` but with 2 sub-centroids per expert
                          (128 total); expert score = max cosine sim over its 2 sub-centroids,
                          M-step updates the winning sub-centroid (winner-takes-all)
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, cast

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import (
    DataMix,
    NumpyDataLoaderConfig,
    NumpyFSLDatasetConfig,
    NumpyPaddedFSLDatasetConfig,
    TokenizerConfig,
)
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe import MoEConfig, MoERouterGatingFunction, MoERouterType
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
from olmo_core.train import (
    Duration,
    TrainerConfig,
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GPUMemoryMonitorCallback,
    LMEvaluatorCallbackConfig,
    ProfilerCallback,
    WandBCallback,
)
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)
from olmo_core.utils import seed_all

SEQUENCE_LENGTH = 2048
GLOBAL_BATCH_SIZE = 1152 * SEQUENCE_LENGTH
MAX_TOKENS = 100_000_000_000

REPO_DIR = os.environ.get("OLMO_CORE_REPO_DIR", str(Path(__file__).resolve().parents[3]))
DATA_ROOT = os.environ.get("OLMO_DATA_ROOT", "dataset/olmoe-1pct")
DATASET_DIR = os.environ.get("OLMO_DATASET_DIR", f"{DATA_ROOT}/tokenized")
EVAL_BASE_DIR = os.environ.get("OLMO_EVAL_BASE_DIR", DATA_ROOT)
WANDB_ENTITY = os.environ.get("WANDB_ENTITY") or None
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "MoE")
WANDB_ENABLED = os.environ.get("WANDB_DISABLED", "0") != "1"

DATA_PATHS = [
    f"{DATASET_DIR}/algebraic-stack/part-0-00000.npy",
    f"{DATASET_DIR}/dclm/part-00-00000.npy",
    f"{DATASET_DIR}/dclm/part-01-00000.npy",
    f"{DATASET_DIR}/dclm/part-02-00000.npy",
    f"{DATASET_DIR}/dclm/part-02-00001.npy",
    f"{DATASET_DIR}/dclm/part-03-00000.npy",
    f"{DATASET_DIR}/dclm/part-04-00000.npy",
    f"{DATASET_DIR}/dclm/part-05-00000.npy",
    f"{DATASET_DIR}/dclm/part-05-00001.npy",
    f"{DATASET_DIR}/dclm/part-06-00000.npy",
    f"{DATASET_DIR}/dclm/part-07-00000.npy",
    f"{DATASET_DIR}/dclm/part-08-00000.npy",
    f"{DATASET_DIR}/dclm/part-09-00000.npy",
    f"{DATASET_DIR}/dclm/part-10-00000.npy",
    f"{DATASET_DIR}/dclm/part-10-00001.npy",
    f"{DATASET_DIR}/dclm/part-11-00000.npy",
    f"{DATASET_DIR}/dclm/part-12-00000.npy",
    f"{DATASET_DIR}/dclm/part-13-00000.npy",
    f"{DATASET_DIR}/dclm/part-13-00001.npy",
    f"{DATASET_DIR}/dclm/part-14-00000.npy",
    f"{DATASET_DIR}/dclm/part-15-00000.npy",
    f"{DATASET_DIR}/open-web-math/part-0-00000.npy",
    f"{DATASET_DIR}/pes2o/part-0-00000.npy",
    f"{DATASET_DIR}/pes2o/part-0-00001.npy",
    f"{DATASET_DIR}/starcoder/part-0-00000.npy",
    f"{DATASET_DIR}/starcoder/part-1-00000.npy",
    f"{DATASET_DIR}/starcoder/part-2-00000.npy",
    f"{DATASET_DIR}/starcoder/part-3-00000.npy",
    f"{DATASET_DIR}/starcoder/part-4-00000.npy",
    f"{DATASET_DIR}/starcoder/part-5-00000.npy",
    f"{DATASET_DIR}/starcoder/part-6-00000.npy",
    f"{DATASET_DIR}/starcoder/part-7-00000.npy",
    f"{DATASET_DIR}/starcoder/part-8-00000.npy",
    f"{DATASET_DIR}/wiki/part-0-00000.npy",
    f"{DATASET_DIR}/wiki/part-0-00001.npy",
]


class RoutingVariant(StrEnum):
    baseline = "baseline"
    deepseek = "deepseek"
    ema = "ema"
    ema_trend = "ema_trend"
    ema_trend_damped = "ema_trend_damped"
    ema_trend_bias = "ema_trend_bias"
    ema_centroid = "ema_centroid"
    ema_centroid_randinit = "ema_centroid_randinit"
    ema_centroid_deepseek = "ema_centroid_deepseek"
    ema_centroid_deepseek_c2 = "ema_centroid_deepseek_c2"
    ema_centroid_sph = "ema_centroid_sph"
    ema_centroid_sph_c2 = "ema_centroid_sph_c2"
    baseline_no_loss = "baseline_no_loss"
    deepseek_v3 = "deepseek_v3"


def configure_routing(moe: MoEConfig, variant: RoutingVariant) -> None:
    """
    Mutate ``moe`` in-place to enable the chosen routing strategy. The default
    ``moe`` config already has ``lb_loss_weight=0.01`` and ``z_loss_weight=0.001``
    from ``llama_like_moe(...)``; variants that disable these set them to None.
    """
    if variant == RoutingVariant.baseline:
        # Standard softmax top-k with Switch-style auxiliary load-balance loss
        # (lb=0.01) AND router z-loss (z=0.001). The classic OLMoE / Mixtral recipe.
        return

    if variant == RoutingVariant.deepseek:
        # arXiv:2408.15664 (§4) — strict "Auxiliary-Loss-Free Load Balancing":
        # sigmoid → bias-shifted top-k → unbiased gather → L1 renorm → bias update
        # by sign(ideal − actual). γ=u=1e-3 per §4.3 ("Update rate"). NO standard
        # lb_loss (the bias rule replaces it — the whole point of the paper),
        # NO router z-loss, and NO complementary seq-aux loss (that comes later
        # in DeepSeek-V3 / arXiv:2412.19437; not part of the 2408 paper recipe).
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.bias_gamma = 1e-3
        moe.router.normalize_expert_weights = 1.0
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.deepseek_v3:
        # Full DeepSeek-V3 recipe (arXiv:2412.19437 §2.1): everything in `deepseek`
        # plus the complementary sequence-wise auxiliary loss (§2.1.2, α=1e-4).
        # That loss penalises within-sequence token concentration independently of
        # the bias rule, which handles cross-batch imbalance. Together they are the
        # complete published V3 load-balancing stack.
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.bias_gamma = 1e-3
        moe.router.normalize_expert_weights = 1.0
        moe.router.seq_aux_loss_weight = 1e-4
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema:
        # Proposed mechanism: per-expert EMA z-normalization of router logits
        # before softmax. Update cadence: once per optimizer step (post_batch),
        # SUM+COUNT-reduced across ranks. NO auxiliary losses, NO router z-loss
        # — the only load-balancing signal is the normalization itself.
        moe.router.ema_zscore_normalize = True
        moe.router.ema_zscore_alpha = 0.99
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_trend:
        # `ema` + Holt's linear-trend (double exponential) smoothing on the MEAN only.
        # Undamped (φ=1 implicit): forecast μ̂ = ℓ + b has zero steady-state lag for
        # linear drift, kills the 2× load imbalance plain `ema` accumulates from its
        # 99-step μ lag. Variance stays plain EMA — Holt's on variance would introduce
        # the E[X²]/μ² cancellation failure mode (see EMA_ZSCORE_ANALYSIS.md next to
        # the router). 1000-step plain-EMA warmup on the trend avoids fitting LR-warmup
        # noise into the slope.
        moe.router.ema_zscore_normalize = True
        moe.router.ema_zscore_alpha = 0.99
        moe.router.ema_zscore_trend = True
        moe.router.ema_zscore_trend_beta = 0.9
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_trend_damped:
        # `ema_trend` with Gardner-McKenzie damped Holt's (φ=0.9). Forecast is
        # μ̂ = ℓ + 0.9·b, which gives a small steady-state lag proportional to
        # (1−φ)·m/σ̂. That lag creates a z-score offset that suppresses the softmax
        # share for drifting experts, acting as an implicit gradient-suppression
        # regularizer on router weight magnitude WITHOUT adding a loss term.
        #
        # Tradeoff vs undamped `ema_trend`:
        # - μ̂ equilibrium magnitude is tighter (~30% smaller at φ=0.9 for high-drift
        #   experts), which keeps router weight norms from drifting as far.
        # - Reintroduces a small routing centering bias (~0.4σ for the most
        #   extreme expert, ~1.5× favoritism) — the cost of the regularization.
        # See EMA_ZSCORE_ANALYSIS.md §4 for the full cost/benefit discussion.
        moe.router.ema_zscore_normalize = True
        moe.router.ema_zscore_alpha = 0.99
        moe.router.ema_zscore_trend = True
        moe.router.ema_zscore_trend_beta = 0.9
        moe.router.ema_zscore_trend_damping = 0.9
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_trend_bias:
        # `ema_trend` + DeepSeek's bias rule (arXiv:2408.15664 §4). z-norm + Holt's
        # handle the first-two-moment stability (mean/scale per expert), the bias
        # rule handles the token-count direction that z-norm is blind to: z-norm
        # equalizes moments 1–2 but not tail shape, so top-k selection count can
        # still drift. `score_bias_e ← score_bias_e + γ·sign(ideal − actual)` is
        # added to scores before top-k (NOT to gather weights), γ=1e-3 per §4.3.
        # Unlike the DeepSeek recipe this keeps SOFTMAX + no L1 renorm — the
        # question is whether bias-rule-for-LI composes with softmax+z-norm+Holt's
        # for a fully aux-loss-free MoE.
        moe.router.ema_zscore_normalize = True
        moe.router.ema_zscore_alpha = 0.99
        moe.router.ema_zscore_trend = True
        moe.router.ema_zscore_trend_beta = 0.9
        moe.router.bias_gamma = 1e-3
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant in (RoutingVariant.ema_centroid, RoutingVariant.ema_centroid_randinit):
        # Primal-dual algorithm for capacity-constrained online clustering (paper §3).
        #
        # Primal (M-step): c_k ← α_t·c_k + (1-α_t)·mean_{assigned}(h)
        #   Step size (1-α_t) = centroid_lr_lambda · η_t is tied to the optimizer LR,
        #   so centroid drift scales with the network's gradient step size and naturally
        #   cools down with the cosine schedule. Zero inter-centroid coupling.
        #
        # Dual (Lagrange multiplier): b_k ← b_k + γ_t·sign(1/K - f_k)
        #   γ_t = bias_lr_lambda · η_t — enforces uniform coverage at the same scale
        #   as the gradient steps, preventing oscillation in the late training regime.
        #
        # Scores: cos(h, c_k) + b_k, top-k selection, identity weights (no softmax).
        # No auxiliary losses, no router weight parameters, no static hyperparameters.
        # (ema_centroid_randinit is a legacy alias — init is always random unit vectors.)
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_lr_lambda = 1.0
        moe.router.gating_function = MoERouterGatingFunction.identity
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_centroid_deepseek:
        # DeepSeek-style gating (arXiv:2408.15664 §4) on top of the centroid algorithm.
        #
        # Routing logits are the raw dot product `h · c_k`, with centroids constrained
        # to the unit sphere (`centroid_spherical=True`) — the analog of weight decay
        # on a learned linear router, so logits stay in std ~1 (instead of std ~34)
        # and sigmoid gives genuine soft weights (instead of saturating to 0/1).
        #
        # Gating + L1 match arXiv:2408.15664 §4:
        #   sigmoid(h · c_k) → bias-shifted top-k → L1 renorm.
        # Bias rule uses constant `bias_gamma=1e-3` (deepseek-style) instead of the
        # LR-tied `bias_lr_lambda` — matches deepseek's load-balancing rate exactly,
        # avoiding the ~10× slower bias accumulation during LR warmup.
        #
        # No seq-aux loss (DeepSeek-V3's complementary loss is redundant with the
        # primal-dual bias rule, and would fight the centroid algorithm's natural
        # within-sequence content clustering).
        #
        # M-step is the spherical k-means update (L2-normalised mean before lerp), to
        # keep centroids on the unit sphere and consistent with the routing geometry.
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_gamma = 1e-3
        moe.router.centroid_spherical = True
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.normalize_expert_weights = 1.0
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_centroid_deepseek_c2:
        # As `ema_centroid_deepseek` but each expert has C=2 sub-centroids (128 unit
        # vectors total). Expert k's routing logit = max over its 2 sub-centroids of
        # `h · c_{k,c}`; M-step updates only the winning sub-centroid (winner-takes-all,
        # picked by cosine similarity in `_accumulate_centroid`). Lets each expert
        # cover two distinct directional modes while keeping the deepseek-style soft
        # sigmoid + L1 gating.
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_gamma = 1e-3
        moe.router.centroid_spherical = True
        moe.router.num_centroids_per_expert = 2
        moe.router.gating_function = MoERouterGatingFunction.sigmoid
        moe.router.normalize_expert_weights = 1.0
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_centroid_sph:
        # Same primal-dual algorithm as `ema_centroid` but with the spherical k-means
        # M-step: the observed cluster mean is L2-normalised before the lerp, keeping
        # centroids on the unit sphere and making the update consistent with the cosine
        # similarity routing criterion. Ablates whether the standard k-means M-step
        # (raw mean) vs. spherical M-step (normalised mean) matters empirically.
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_lr_lambda = 1.0
        moe.router.centroid_spherical = True
        moe.router.gating_function = MoERouterGatingFunction.identity
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.ema_centroid_sph_c2:
        # As `ema_centroid_sph` but each expert has C=2 sub-centroids (128 total vectors).
        # Expert score = max cosine sim over its 2 sub-centroids; M-step updates only the
        # winning sub-centroid (winner-takes-all). Lets each expert cover two distinct modes.
        moe.router.name = MoERouterType.centroid
        moe.router.centroid_lr_lambda = 10.0
        moe.router.bias_lr_lambda = 1.0
        moe.router.centroid_spherical = True
        moe.router.num_centroids_per_expert = 2
        moe.router.gating_function = MoERouterGatingFunction.identity
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    if variant == RoutingVariant.baseline_no_loss:
        # Softmax top-k with no auxiliary losses — pure routing signal, no lb_loss,
        # no z_loss. Floor for what load imbalance looks like without any constraint.
        moe.lb_loss_weight = None
        moe.z_loss_weight = None
        return

    raise ValueError(f"unknown routing variant: {variant!r}")


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198


def build_config(run_name: str, routing: RoutingVariant, overrides: List[str]) -> ExperimentConfig:
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    d_model = 1024

    model_config = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=9,
        n_heads=8,
        num_experts=64,
        top_k=6,
        expert_hidden_size=int(0.5 * d_model),
        # = 2 × expert_hidden_size: equivalent to N_s=2 shared experts at expert
        # size (DeepSeek aux-loss-free 1B, arXiv:2408.15664 §4 Table 5). olmo-core's
        # MoEConfig only supports one shared MLP, so we fold N_s=2 into one wider one.
        shared_expert_hidden_size=2 * int(0.5 * d_model),
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
        # init_std=0.02 — "Xavier/He-ish" (≈sqrt(2/(5·d_model)) at d=1024), the
        # standard OLMoE/Switch/Megatron MoE baseline. The aux-loss-free paper
        # (arXiv:2408.15664 App. B) uses 0.006, but that's a DeepSeek house-style
        # constant reused verbatim across all their scales (DeepSeekMoE 2B/16B/145B)
        # rather than fan-in-scaled. At d=1024 it's ~3× smaller than the Xavier/He
        # rule, leaving router logits near zero — an empirical confounder: all
        # three variants got stuck at max/mean load-imbalance ≈9 regardless of
        # routing strategy. 0.02 lets the softmax baseline behave as intended,
        # and since we share init across variants the comparison stays fair.
        init_std=0.02,
    )

    block = cast(TransformerBlockConfig, model_config.block)
    moe = block.feed_forward_moe
    assert moe is not None, "expected an MoE block from llama_like_moe"
    configure_routing(moe, routing)

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=4096,
        tokenizer=tokenizer,
        work_dir=f"{REPO_DIR}/dataset-cache",
    )

    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=6198,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=96 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=SkipStepAdamWConfig(
            lr=1e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
        ),
        compile_model=True,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        z_loss_multiplier=1e-5,
        max_grad_norm=1.0,
        # alpha_f=0.1 default → min_lr = 1e-4 (matches paper's 1e-3 → 1e-4 cosine).
        scheduler=CosWithWarmup(warmup=1000),
    )

    # Tag the WandB run with the routing variant so charts can be grouped.
    wandb_name = f"{run_name}-{routing.value}"

    trainer_config = (
        TrainerConfig(
            save_folder=f"{REPO_DIR}/runs/{wandb_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            max_duration=Duration.tokens(MAX_TOKENS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=250,
                ephemeral_save_interval=200,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=wandb_name,
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                group=routing.value,
                enabled=WANDB_ENABLED,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
        .with_callback(
            "lm_evaluator",
            LMEvaluatorCallbackConfig(
                eval_dataset=NumpyPaddedFSLDatasetConfig.from_data_mix(
                    DataMix.v3_small_ppl_validation,
                    mix_base_dir=EVAL_BASE_DIR,
                    sequence_length=SEQUENCE_LENGTH,
                    tokenizer=tokenizer,
                    work_dir=f"{REPO_DIR}/dataset-cache",
                ),
                eval_interval=250,
            ),
        )
    )

    return ExperimentConfig(
        model=model_config,
        dataset=dataset_config,
        data_loader=data_loader_config,
        train_module=train_module_config,
        trainer=trainer_config,
    ).merge(overrides)


def main(run_name: str, routing: RoutingVariant, overrides: List[str]):
    config = build_config(run_name, routing, overrides)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


def _parse_routing(args: List[str]) -> tuple[RoutingVariant, List[str]]:
    """Pull ``--routing=VARIANT`` out of argv; return (variant, remaining args)."""
    variant: RoutingVariant | None = None
    remaining: List[str] = []
    for arg in args:
        if arg.startswith("--routing="):
            variant = RoutingVariant(arg.split("=", 1)[1])
        else:
            remaining.append(arg)
    if variant is None:
        valid = ", ".join(v.value for v in RoutingVariant)
        raise SystemExit(f"--routing=VARIANT is required; one of: {valid}")
    return variant, remaining


if __name__ == "__main__":
    if len(sys.argv) < 2:
        valid = ", ".join(v.value for v in RoutingVariant)
        print(
            f"Usage: python {sys.argv[0]} RUN_NAME --routing=VARIANT [OVERRIDES...]\n"
            f"  VARIANT: {valid}"
        )
        sys.exit(1)

    run_name, *rest = sys.argv[1:]
    routing, overrides = _parse_routing(rest)

    prepare_training_environment()
    main(run_name, routing=routing, overrides=overrides)
    teardown_training_environment()
