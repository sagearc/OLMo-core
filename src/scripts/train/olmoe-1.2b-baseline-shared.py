"""
OLMoE 1.2B baseline with shared expert, matching the original olmoe-rocm experiment.

Architecture: d_model=2048, n_heads=16, n_layers=8, 8 experts, top-2, dropless, shared expert.
Data: local 1%-sample dataset at /home/morg/students/sagiahrac/dataset/olmoe-1pct/tokenized/

Usage:
    python src/scripts/train/olmoe-1.2b-baseline-shared.py RUN_NAME [OVERRIDES...]
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup, OptimGroupOverride
from olmo_core.train import (
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

SEQUENCE_LENGTH = 4096
GLOBAL_BATCH_SIZE = 1024 * SEQUENCE_LENGTH

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


@dataclass
class ExperimentConfig(Config):
    model: TransformerConfig
    dataset: NumpyDatasetConfig
    data_loader: NumpyDataLoaderConfig
    train_module: TransformerTrainModuleConfig
    trainer: TrainerConfig
    init_seed: int = 6198


def build_config(run_name: str, overrides: List[str]) -> ExperimentConfig:
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    d_model = 2048

    model_config = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=8,
        n_heads=16,
        num_experts=8,
        top_k=2,
        expert_hidden_size=int(0.5 * d_model),
        # Equivalent to DeepSeek's N_s=2 shared experts at expert size
        # (arXiv:2408.15664 §4 Table 5). olmo-core's MoEConfig only supports a single
        # shared MLP, so we fold N_s=2 into one shared MLP at 2× expert size.
        shared_expert_hidden_size=2 * int(0.5 * d_model),
        dropless=True,
        reordered_norm=True,
        qk_norm=True,
        rope_theta=500_000,
        layer_norm_eps=1e-6,
        lb_loss_weight=0.01,
        z_loss_weight=0.001,
    )

    dataset_config = NumpyFSLDatasetConfig(
        paths=DATA_PATHS,
        sequence_length=SEQUENCE_LENGTH,
        max_target_sequence_length=8192,
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
            lr=4e-4,
            weight_decay=0.1,
            betas=(0.9, 0.95),
            group_overrides=[
                OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
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
        scheduler=CosWithWarmup(warmup_steps=256),
    )

    trainer_config = (
        TrainerConfig(
            save_folder=f"{REPO_DIR}/runs/{run_name}",
            save_overwrite=True,
            metrics_collect_interval=10,
            cancel_check_interval=1,
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
                name=run_name,
                entity="sagiah",
                project="MoE",
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


def main(run_name: str, overrides: List[str]):
    config = build_config(run_name, overrides)

    seed_all(config.init_seed)

    model = config.model.build(init_device="meta")
    train_module = config.train_module.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(dataset, dp_process_group=train_module.dp_process_group)
    trainer = config.trainer.build(train_module, data_loader)

    config_dict = config.as_config_dict()
    cast(ConfigSaverCallback, trainer.callbacks["config_saver"]).config = config_dict

    trainer.fit()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} RUN_NAME [OVERRIDES...]")
        sys.exit(1)

    run_name, *overrides = sys.argv[1:]

    prepare_training_environment()
    main(run_name, overrides=overrides)
    teardown_training_environment()
