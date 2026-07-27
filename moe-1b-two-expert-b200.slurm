#!/bin/bash
#SBATCH --job-name=moe1b-2e
#SBATCH --output=/data/sagiahrac/moe1b-two-expert-forward-loo/logs/%x-%j.out
#SBATCH --error=/data/sagiahrac/moe1b-two-expert-forward-loo/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu-b200-killable
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail

: "${ROUTING:?Set ROUTING to deepseek or deepseek_binary_leave_one_out}"
: "${RUN_TAG:?Set the shared stable run tag}"
: "${MICROBATCH_SEQUENCES:=96}"
: "${GLOBAL_BATCH_SEQUENCES:=1152}"
: "${SMOKE_STEPS:=0}"
: "${ONLINE:=1}"

case "${ROUTING}" in
    deepseek|deepseek_binary_leave_one_out) ;;
    *)
        echo "Invalid ROUTING=${ROUTING}" >&2
        exit 2
        ;;
esac
if ((GLOBAL_BATCH_SEQUENCES % MICROBATCH_SEQUENCES != 0)); then
    echo "MICROBATCH_SEQUENCES must divide GLOBAL_BATCH_SEQUENCES" >&2
    exit 2
fi

ROOT=/data/sagiahrac/moe1b-two-expert-forward-loo
SOURCE="${ROOT}/source-2a25109"
PYTHON=/data/sagiahrac/vllm-b200/.venv/bin/python
DATA=/data/sagiahrac/fig9-half-b200/data/wiki/part-0-00001.npy
RUN_NAME="${RUN_TAG}-${ROUTING}"
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
RANK_MICROBATCH_SIZE=$((MICROBATCH_SEQUENCES * 2048))
GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SEQUENCES * 2048))

test -f "${SOURCE}/src/scripts/train/moe-1b.py"
test -f "${DATA}"
mkdir -p "${RUN_DIR}" "${ROOT}/cache/${SLURM_JOB_ID}"

export PYTHONPATH="${ROOT}/site:/data/sagiahrac/vllm-b200/site:${SOURCE}/src"
export PYTHONUNBUFFERED=1
export OLMO_CORE_USE_TORCH_GROUPED_MM=1
export XDG_CACHE_HOME="${ROOT}/cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

wandb_args=()
if ((ONLINE)); then
    : "${WANDB_API_KEY:?WANDB_API_KEY is required for ONLINE=1}"
    export WANDB_MODE=online
    export WANDB_RESUME=allow
    export WANDB_RUN_ID="${RUN_NAME}"
else
    export WANDB_MODE=disabled
    wandb_args+=("--trainer.callbacks.wandb.enabled=false")
fi

hard_stop_args=()
if ((SMOKE_STEPS > 0)); then
    hard_stop_args+=("--trainer.hard_stop={unit: steps, value: ${SMOKE_STEPS}}")
fi

echo "host=$(hostname)"
echo "routing=${ROUTING}"
echo "run_name=${RUN_NAME}"
echo "architecture=d1024-l9-h8-experts2-top1"
echo "microbatch_sequences=${MICROBATCH_SEQUENCES}"
echo "global_batch_sequences=${GLOBAL_BATCH_SEQUENCES}"
echo "gradient_accumulation_steps=$((GLOBAL_BATCH_SEQUENCES / MICROBATCH_SEQUENCES))"
sha256sum \
    "${SOURCE}/src/olmo_core/nn/moe/router.py" \
    "${SOURCE}/src/scripts/train/moe-1b.py"

"${PYTHON}" -u \
    "${SOURCE}/src/scripts/train/moe-1b.py" \
    "${RUN_NAME}" \
    "--routing=${ROUTING}" \
    "--model.block.feed_forward_moe.num_experts=2" \
    "--model.block.feed_forward_moe.router.top_k=1" \
    "--model.block.feed_forward_moe.router.normalize_expert_weights=null" \
    "--dataset.paths=[${DATA}]" \
    "--dataset.work_dir=${ROOT}/dataset-cache" \
    "--data_loader.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "--data_loader.num_workers=0" \
    "--train_module.rank_microbatch_size=${RANK_MICROBATCH_SIZE}" \
    "--train_module.compile_model=false" \
    "--trainer.save_folder=${RUN_DIR}" \
    "--trainer.metrics_collect_interval=10" \
    "--trainer.callbacks.wandb.group=moe1b-two-expert-forward-loo" \
    "--trainer.callbacks.checkpointer.save_interval=100" \
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=25" \
    "--trainer.callbacks.lm_evaluator.enabled=false" \
    "--trainer.no_evals" \
    "${wandb_args[@]}" \
    "${hard_stop_args[@]}"
