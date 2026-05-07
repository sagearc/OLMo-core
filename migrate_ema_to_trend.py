"""
Migrate an old ema_trend checkpoint (buffers: _ema_sq, _ema_sq_trend, _ema_mean_trend)
to the new ema_trend code (buffers: _ema_var, _ema_mean_trend).

Transformations per MoE router block:
  _ema_sq       → drop; compute _ema_var = _ema_sq - _ema_mean² / (1 - α^step)
                  preserves σ̂² at the moment of load; new EMA takes over
                  from that starting point.
  _ema_sq_trend → drop (new code has no sq/var trend buffer; Holt's is mean-only)
  _ema_mean_trend → keep as-is (already zero at step == warmup)

Usage:
    .venv/bin/python migrate_ema_to_trend.py
"""

import io
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader, RemoteFileSystemWriter

SRC = Path("runs/moe-1b-278888-ema_trend/step1000")
DST = Path("runs/moe-1b-278888-ema_trend/step1000-new-code-compat")

N_BLOCKS = 9
NUM_EXPERTS = 64
ALPHA = 0.99
STEP = 1000


def build_state_dict_template(reader: RemoteFileSystemReader) -> dict:
    metadata = reader.read_metadata()
    sd = {}
    for key, meta in metadata.state_dict_metadata.items():
        if isinstance(meta, TensorStorageMetadata):
            sd[key] = torch.zeros(meta.size, dtype=meta.properties.dtype)
        else:
            sd[key] = io.BytesIO()
    return sd


def main():
    src_model_optim = SRC / "model_and_optim"
    dst_model_optim = DST / "model_and_optim"

    print(f"Loading checkpoint from {src_model_optim} ...")
    reader = RemoteFileSystemReader(str(src_model_optim))
    state_dict = build_state_dict_template(reader)
    dist_cp.load(state_dict, checkpoint_id=str(src_model_optim), storage_reader=reader)
    print(f"  Loaded {len(state_dict)} keys.")

    # Verify step matches expectation before dividing by bias-correction.
    for i in range(N_BLOCKS):
        step_key = f"model.blocks.{i}.feed_forward_moe.router._ema_step_count"
        actual = int(state_dict[step_key].item())
        assert actual == STEP, f"Block {i}: expected step={STEP}, got {actual}. Update STEP."

    bc = 1.0 - ALPHA**STEP

    for i in range(N_BLOCKS):
        prefix = f"model.blocks.{i}.feed_forward_moe.router"
        sq_key = f"{prefix}._ema_sq"
        sq_trend_key = f"{prefix}._ema_sq_trend"
        mean_key = f"{prefix}._ema_mean"
        var_key = f"{prefix}._ema_var"

        assert sq_key in state_dict, f"Missing {sq_key}"
        assert sq_trend_key in state_dict, f"Missing {sq_trend_key}"
        assert var_key not in state_dict, f"{var_key} already present — already migrated?"

        ema_sq = state_dict.pop(sq_key)
        state_dict.pop(sq_trend_key)  # drop: new code has no sq/var trend buffer
        ema_mean = state_dict[mean_key]

        # Preserve σ̂²: old formula was (ema_sq/bc - (ema_mean/bc)²)
        # new _ema_var satisfies: _ema_var/bc = same value → _ema_var = ema_sq - ema_mean²/bc
        ema_var = ema_sq - ema_mean.pow(2) / bc
        ema_var.clamp_(min=0.0)  # guard against fp noise giving tiny negatives
        state_dict[var_key] = ema_var.to(torch.float32)

    print(f"  Transformed {N_BLOCKS} blocks: _ema_sq→_ema_var (σ̂² preserved), _ema_sq_trend dropped.")

    print(f"Saving migrated checkpoint to {dst_model_optim} ...")
    dst_model_optim.mkdir(parents=True, exist_ok=True)
    dist_cp.save(state_dict, storage_writer=RemoteFileSystemWriter(str(dst_model_optim)))
    print("  Saved.")

    for name in ("config.json", "data_paths.txt"):
        src_file = SRC / name
        if src_file.exists():
            shutil.copy(src_file, DST / name)
            print(f"  Copied {name}")

    train_src = SRC / "train"
    train_dst = DST / "train"
    if train_src.exists():
        shutil.copytree(train_src, train_dst, dirs_exist_ok=True)
        print(f"  Copied train/")

    print(f"\nDone.  Point trainer.load_path to:\n  {DST.resolve()}")


if __name__ == "__main__":
    main()
