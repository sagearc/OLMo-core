#!/bin/bash
#SBATCH --job-name=fig9-half-b200
#SBATCH --output=/data/sagiahrac/fig9-half-b200/logs/%x-%j.out
#SBATCH --error=/data/sagiahrac/fig9-half-b200/logs/%x-%j.err
#SBATCH --partition=gpu-b200-killable
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-b200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:b200:1
#SBATCH --time=00:20:00

set -euo pipefail

: "${ROUTING:?Set ROUTING to deepseek or leave_one_out}"
: "${RUN_TAG:=fig9-half-b200-smoke}"
: "${STEPS:=10}"
: "${MICROBATCH_SEQUENCES:=64}"
: "${BIAS_GAMMA:=0.001}"
: "${BIAS_LR_LAMBDA:=null}"
: "${ONLINE:=0}"

case "$ROUTING" in
    deepseek|leave_one_out) ;;
    *)
        echo "ROUTING must be deepseek or leave_one_out, got: $ROUTING" >&2
        exit 2
        ;;
esac
if ((256 % MICROBATCH_SEQUENCES != 0)); then
    echo "MICROBATCH_SEQUENCES must divide the 256-sequence global batch" >&2
    exit 2
fi

ROOT=/data/sagiahrac/fig9-half-b200
SOURCE="${ROOT}/source"
PYTHON=/data/sagiahrac/vllm-b200/.venv/bin/python
DATA="${ROOT}/data/wiki/part-0-00001.npy"
RUN_NAME="${RUN_TAG}-${ROUTING}-${SLURM_JOB_ID}"
RUN_DIR="${ROOT}/runs/${RUN_NAME}"
RANK_MICROBATCH_SIZE=$((MICROBATCH_SEQUENCES * 2048))

test -f "${SOURCE}/src/scripts/train/olmoe-figure9-half-router.py"
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
    : "${WANDB_API_KEY:?ONLINE=1 requires WANDB_API_KEY}"
    export WANDB_MODE=online
else
    export WANDB_MODE=disabled
    wandb_args+=("--trainer.callbacks.wandb.enabled=false")
fi

echo "host=$(hostname)"
echo "gpu=${CUDA_VISIBLE_DEVICES:-unset}"
echo "routing=${ROUTING}"
echo "steps=${STEPS}"
echo "global_batch_sequences=256"
echo "microbatch_sequences=${MICROBATCH_SEQUENCES}"
echo "gradient_accumulation_steps=$((256 / MICROBATCH_SEQUENCES))"
echo "bias_gamma=${BIAS_GAMMA}"
echo "bias_lr_lambda=${BIAS_LR_LAMBDA}"
echo "wandb_mode=${WANDB_MODE}"
sha256sum \
    "${SOURCE}/src/olmo_core/nn/moe/router.py" \
    "${SOURCE}/src/scripts/train/olmoe-figure9-half-router.py"

"${PYTHON}" -u \
    "${SOURCE}/src/scripts/train/olmoe-figure9-half-router.py" \
    "${RUN_NAME}" \
    "--routing=${ROUTING}" \
    "--dataset.paths=[${DATA}]" \
    "--dataset.work_dir=${ROOT}/dataset-cache" \
    "--model.block.feed_forward_moe.router.bias_gamma=${BIAS_GAMMA}" \
    "--model.block.feed_forward_moe.router.bias_lr_lambda=${BIAS_LR_LAMBDA}" \
    "--data_loader.num_workers=0" \
    "--train_module.rank_microbatch_size=${RANK_MICROBATCH_SIZE}" \
    "--train_module.compile_model=false" \
    "--trainer.save_folder=${RUN_DIR}" \
    "--trainer.metrics_collect_interval=1" \
    "${wandb_args[@]}" \
    "--trainer.callbacks.lm_evaluator.enabled=false" \
    "--trainer.no_evals" \
    "--trainer.no_checkpoints" \
    "--trainer.hard_stop={unit: steps, value: ${STEPS}}"
