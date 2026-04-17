"""
OLMoE 1.2B with DeepSeek-v3 auxiliary-loss-free routing (arXiv:2412.19437 §2.1.2).

Same shape as ``olmoe-1.2b-baseline-shared.py`` (d_model=2048, n_layers=8, 8 experts,
top-2, dropless, shared expert), but with the routing strategy swapped:
  - sigmoid per-expert gating
  - per-expert bias added to scores for top-k selection only (gamma=1e-3)
  - gate weights = unbiased sigmoid scores, L1-renormalized
  - bias updated each optimizer step via b += gamma * sign(expected - actual)
  - complementary sequence-wise auxiliary loss with weight alpha=1e-4
    (microbatch-wise approximation; see deepseek_seq_aux_loss docstring)
  - standard load-balancing loss disabled (replaced by the bias rule)
  - z-loss kept at 1e-3 for logit-magnitude stability

Usage:
    python src/scripts/train/olmoe-1.2b-deepseek-shared.py RUN_NAME [OVERRIDES...]
"""

import sys
from dataclasses import dataclass
from typing import List, cast

from olmo_core.config import Config, DType
from olmo_core.data import NumpyDataLoaderConfig, NumpyFSLDatasetConfig, TokenizerConfig
from olmo_core.data.numpy_dataset import NumpyDatasetConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
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

    block = cast(TransformerBlockConfig, model_config.block)
    moe = block.feed_forward_moe
    assert moe is not None, "expected an MoE block from llama_like_moe"
    moe.router.gating_function = MoERouterGatingFunction.sigmoid
    moe.router.bias_gamma = 1e-3
    moe.router.normalize_expert_weights = 1.0
    moe.router.seq_aux_loss_weight = 1e-4
    # Disable both auxiliary losses: the bias rule replaces lb_loss, and router_z_loss
    # (logsumexp(logits)^2) is a softmax-shaped penalty with no probabilistic meaning
    # under sigmoid gating — DeepSeek-V3 does not use it.
    moe.lb_loss_weight = None
    moe.z_loss_weight = None
    # NOTE: HF/DeepSeek apply a `routed_scaling_factor` (=2.5 for 671B). Omitted here:
    # at top_k=2 of 8 experts the L1-renormalized weights already sum to 1, so the
    # output-magnitude collapse that motivates the multiplier at 8-of-256 sparsity
    # does not arise. Revisit if scaling experts.

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
        compile_model=False,
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
