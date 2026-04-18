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
        baseline   softmax + Switch lb_loss (0.01) + router z-loss (0.001) — floor
        deepseek   sigmoid + bias rule (γ=1e-3) — strict arXiv:2408.15664, no aux loss
        ema        EMA z-norm + softmax (proposed) — no aux loss, no z-loss
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType, StrEnum
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe import MoEConfig, MoERouterGatingFunction
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
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

DATASET_DIR = "/home/morg/students/sagiahrac/dataset/olmoe-1pct/tokenized"
REPO_DIR = "/home/morg/students/sagiahrac/repos/olmo-core-repo"

DATA_PATHS = [
    f"{DATASET_DIR}/algebraic-stack/part-0-00000.npy",
    f"{DATASET_DIR}/dclm/part-00-00000.npy",
    f"{DATASET_DIR}/dclm/part-01-00000.npy",
    f"{DATASET_DIR}/dclm/part-02-00000.npy",
    f"{DATASET_DIR}/dclm/part-03-00000.npy",
    f"{DATASET_DIR}/dclm/part-04-00000.npy",
    f"{DATASET_DIR}/dclm/part-05-00000.npy",
    f"{DATASET_DIR}/dclm/part-06-00000.npy",
    f"{DATASET_DIR}/dclm/part-07-00000.npy",
    f"{DATASET_DIR}/dclm/part-08-00000.npy",
    f"{DATASET_DIR}/dclm/part-09-00000.npy",
    f"{DATASET_DIR}/dclm/part-10-00000.npy",
    f"{DATASET_DIR}/dclm/part-11-00000.npy",
    f"{DATASET_DIR}/dclm/part-12-00000.npy",
    f"{DATASET_DIR}/dclm/part-13-00000.npy",
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
        rank_microbatch_size=32 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=1e-3,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
            ],
            fused=True,
        ),
        compile_model=False,
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
                save_interval=1000,
                ephemeral_save_interval=200,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=wandb_name,
                entity="sagiah",
                project="MoE",
                group=routing.value,
                enabled=True,
                cancel_check_interval=10,
            ),
        )
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback("profiler", ProfilerCallback(enabled=False))
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
