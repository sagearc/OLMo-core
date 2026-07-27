"""
Matched small-router ablation based on the exact Figure 9 runs from OLMoE
(arXiv:2409.02060).

Both variants use the same 8-layer, width-512, 8-expert, top-4 model and the
DeepSeek auxiliary-loss-free sigmoid router. They differ only in which half of
the trainable router rows receives the routing-weight gradient.
"""

import sys
from dataclasses import dataclass
from typing import cast

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
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.moe import MoERouterGatingFunction, MoERouterType
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
GLOBAL_BATCH_SIZE = 256 * SEQUENCE_LENGTH
MAX_STEPS = 19_074
INIT_SEED = 124

DATASET_DIR = "/home/morg/students/sagiahrac/dataset/olmoe-1pct/tokenized"
EVAL_BASE_DIR = "/home/morg/students/sagiahrac/dataset/olmoe-1pct"
REPO_DIR = "/home/morg/students/sagiahrac/repos/olmo-core-repo"

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
    deepseek = "deepseek"
    leave_one_out = "leave_one_out"


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = INIT_SEED


def build_config(run_name: str, routing: RoutingVariant, overrides: list[str]) -> ExperimentConfig:
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    d_model = 512

    model_config = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=8,
        n_heads=8,
        num_experts=8,
        top_k=4,
        # Old Figure 9 expert: 2 * 512 * 2048 parameters.
        # Current gated expert: 3 * 512 * 1368 parameters (0.195% more).
        # 1368 is the nearest BF16 grouped-GEMM-compatible multiple of 8.
        expert_hidden_size=1368,
        shared_expert_hidden_size=None,
        dropless=True,
        reordered_norm=False,
        qk_norm=False,
        rope_theta=10_000,
        layer_norm_eps=1e-5,
        lb_loss_weight=None,
        z_loss_weight=None,
        init_std=0.02,
    )

    layer_norm = LayerNormConfig(
        name=LayerNormType.default,
        eps=1e-5,
        elementwise_affine=False,
        bias=False,
    )
    block = cast(TransformerBlockConfig, model_config.block)
    block.layer_norm = layer_norm
    model_config.lm_head.layer_norm = layer_norm

    moe = block.feed_forward_moe
    assert moe is not None
    moe.router.gating_function = MoERouterGatingFunction.sigmoid
    moe.router.bias_gamma = 1e-3
    moe.router.normalize_expert_weights = 1.0
    if routing == RoutingVariant.leave_one_out:
        moe.router.name = MoERouterType.half_leave_one_out

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        work_dir=f"{REPO_DIR}/dataset-cache",
    )
    data_loader_config = NumpyDataLoaderConfig(
        global_batch_size=GLOBAL_BATCH_SIZE,
        seed=INIT_SEED,
        num_workers=4,
    )

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=64 * SEQUENCE_LENGTH,
        max_sequence_length=SEQUENCE_LENGTH,
        optim=AdamWConfig(
            lr=2e-4,
            eps=1e-8,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts={"weight_decay": 0.0})
            ],
            fused=True,
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
        scheduler=CosWithWarmup(warmup=191),
    )

    wandb_name = f"{run_name}-{routing.value}"
    trainer_config = (
        TrainerConfig(
            save_folder=f"{REPO_DIR}/runs/{wandb_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
            max_duration=Duration.steps(MAX_STEPS),
        )
        .with_callback("gpu_monitor", GPUMemoryMonitorCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=500,
                ephemeral_save_interval=250,
                save_async=True,
            ),
        )
        .with_callback(
            "wandb",
            WandBCallback(
                name=wandb_name,
                entity="MoEPaper",
                project="MoE",
                group="olmoe-figure9-half-router",
                enabled=True,
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
                eval_interval=500,
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


def main(run_name: str, routing: RoutingVariant, overrides: list[str]) -> None:
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


def _parse_routing(args: list[str]) -> tuple[RoutingVariant, list[str]]:
    variant: RoutingVariant | None = None
    remaining: list[str] = []
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
