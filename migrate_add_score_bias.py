"""
One-off migration: take the existing step-1000 migrated checkpoint
(runs/moe-1b-278888-ema_trend/step1000-new-code-compat) and add zero
`score_bias` buffers per MoE router block so it can be loaded into a
variant with `bias_gamma` set (which registers the buffer).

Output: runs/moe-1b-278888-ema_trend/step1000-bias-compat
"""

import io
import shutil
from pathlib import Path

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

from olmo_core.distributed.checkpoint.filesystem import RemoteFileSystemReader, RemoteFileSystemWriter

SRC = Path("runs/moe-1b-278888-ema_trend/step1000-new-code-compat")
DST = Path("runs/moe-1b-278888-ema_trend/step1000-bias-compat")

N_BLOCKS = 9
NUM_EXPERTS = 64


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

    for i in range(N_BLOCKS):
        key = f"model.blocks.{i}.feed_forward_moe.router.score_bias"
        assert key not in state_dict, f"{key} already present — already migrated?"
        state_dict[key] = torch.zeros(NUM_EXPERTS, dtype=torch.float32)

    print(f"  Injected {N_BLOCKS} zero score_bias buffers.")

    print(f"Saving migrated checkpoint to {dst_model_optim} ...")
    dst_model_optim.mkdir(parents=True, exist_ok=True)
    dist_cp.save(state_dict, storage_writer=RemoteFileSystemWriter(str(dst_model_optim)))
    print("  Saved.")

    for name in ("config.json", "data_paths.txt", ".metadata.json"):
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
