"""
One-shot verification: FSDP2 with `param_dtype=bfloat16` must NOT cast our fp32 EMA
buffers to bf16. Mirrors `score_bias` (which has the same setup and works today).

Run with: torchrun --nproc-per-node=1 src/scripts/train/_verify_ema_fsdp_dtype.py
"""

from typing import cast

import torch

from olmo_core.config import DType
from olmo_core.data import TokenizerConfig
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.nn.moe import MoERouterGatingFunction
from olmo_core.nn.transformer import TransformerBlockConfig, TransformerConfig
from olmo_core.optim import AdamWConfig, CosWithWarmup
from olmo_core.train import prepare_training_environment, teardown_training_environment
from olmo_core.train.train_module import (
    TransformerDataParallelConfig,
    TransformerDataParallelWrappingStrategy,
    TransformerTrainModuleConfig,
)


def main():
    tokenizer = TokenizerConfig.gpt_neox_olmo_dolma_v1_5()
    d_model = 256

    model_config = TransformerConfig.llama_like_moe(
        d_model=d_model,
        vocab_size=tokenizer.padded_vocab_size(),
        n_layers=2,
        n_heads=8,
        num_experts=4,
        top_k=2,
        expert_hidden_size=d_model,
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
    assert moe is not None
    moe.router.gating_function = MoERouterGatingFunction.softmax
    moe.router.ema_zscore_normalize = True
    moe.router.ema_zscore_alpha = 0.99
    moe.router.bias_gamma = 1e-3  # also enable score_bias to compare
    moe.lb_loss_weight = None
    moe.z_loss_weight = None

    train_module_config = TransformerTrainModuleConfig(
        rank_microbatch_size=2 * 64,
        max_sequence_length=64,
        optim=AdamWConfig(lr=4e-4, betas=(0.9, 0.95), fused=True),
        compile_model=False,
        dp_config=TransformerDataParallelConfig(
            name=DataParallelType.fsdp,
            param_dtype=DType.bfloat16,
            reduce_dtype=DType.float32,
            wrapping_strategy=TransformerDataParallelWrappingStrategy.full,
        ),
        max_grad_norm=1.0,
        scheduler=CosWithWarmup(warmup_steps=10),
    )

    model = model_config.build(init_device="meta")
    train_module = train_module_config.build(model)

    print("\n=== FSDP-wrapped buffer dtypes ===")
    for layer_idx, block_mod in enumerate(train_module.model.blocks.values()):  # type: ignore
        moe_mod = block_mod.feed_forward_moe
        router = moe_mod.router
        print(f"layer {layer_idx}:")
        print(
            f"  _ema_mean.dtype  = {router._ema_mean.dtype}, shape={tuple(router._ema_mean.shape)}, device={router._ema_mean.device}"
        )
        print(
            f"  _ema_sq.dtype    = {router._ema_sq.dtype}, shape={tuple(router._ema_sq.shape)}, device={router._ema_sq.device}"
        )
        print(
            f"  score_bias.dtype = {router.score_bias.dtype}, shape={tuple(router.score_bias.shape)}, device={router.score_bias.device}"
        )

    print("\n=== Initial values (post-FSDP materialization) — should be (0, 1) and 0 ===")
    for layer_idx, block_mod in enumerate(train_module.model.blocks.values()):  # type: ignore
        router = block_mod.feed_forward_moe.router
        print(f"layer {layer_idx}:")
        print(f"  _ema_mean    : abs_max={router._ema_mean.abs().max().item():.6e} (expect 0)")
        print(f"  _ema_sq      : value={router._ema_sq.tolist()} (expect ones)")
        print(f"  score_bias   : abs_max={router.score_bias.abs().max().item():.6e} (expect 0)")

    print("\n=== Running one forward + post_batch directly on the wrapped model ===")
    train_module.model.train()
    input_ids = torch.randint(
        0, tokenizer.padded_vocab_size(), (1, 64), device=torch.cuda.current_device()
    )
    _ = train_module.model(input_ids=input_ids)
    train_module.model.post_batch()

    print("\n=== Buffer dtypes AFTER post_batch ===")
    for layer_idx, block_mod in enumerate(train_module.model.blocks.values()):  # type: ignore
        router = block_mod.feed_forward_moe.router
        print(f"layer {layer_idx}:")
        print(
            f"  _ema_mean: dtype={router._ema_mean.dtype}, value_max={router._ema_mean.abs().max().item():.6e}"
        )
        print(
            f"  _ema_sq:   dtype={router._ema_sq.dtype}, value_max={router._ema_sq.abs().max().item():.6e}"
        )
        print(
            f"  score_bias: dtype={router.score_bias.dtype}, value_max={router.score_bias.abs().max().item():.6e}"
        )

    print("\n=== State_dict roundtrip check ===")
    sd = train_module.model.state_dict()
    ema_keys = [k for k in sd.keys() if "_ema_mean" in k or "_ema_sq" in k]
    print(f"EMA keys in state_dict: {len(ema_keys)}")
    for k in ema_keys[:4]:
        print(f"  {k}: dtype={sd[k].dtype}, shape={tuple(sd[k].shape)}")


if __name__ == "__main__":
    prepare_training_environment()
    try:
        main()
    finally:
        teardown_training_environment()
