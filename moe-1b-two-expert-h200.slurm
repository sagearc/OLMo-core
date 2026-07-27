#!/bin/bash
#SBATCH --job-name=moe1b-2e
#SBATCH --output=slurm_output/%x-h200-%j.out
#SBATCH --error=slurm_output/%x-h200-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu-h200-killable
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --requeue

set -euo pipefail

: "${ROUTING:?Set ROUTING to deepseek or deepseek_binary_leave_one_out}"
: "${RUN_TAG:?Set the stable tag shared by the pair}"
: "${SOURCE_DIR:?Set SOURCE_DIR to the immutable source snapshot}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
: "${MICROBATCH_SEQUENCES:=32}"
: "${GLOBAL_BATCH_SEQUENCES:=1152}"

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

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
LOCAL_ROOT=/tmp/sagiahrac-moe1b-two-expert
LOCAL_SOURCE="${LOCAL_ROOT}/source-2a25109"
LOCAL_DATA="${LOCAL_ROOT}/data/wiki-part-0-00001.npy"
NFS_DATA=/home/morg/students/sagiahrac/dataset/olmoe-1pct/tokenized/wiki/part-0-00001.npy
PYTHON="${SUBMIT_DIR}/.venv-cuda/bin/python"
RUN_NAME="${RUN_TAG}"
RUN_ID="${RUN_TAG}-${ROUTING}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/runs/${RUN_ID}"
PERSISTENT_ROOT="/home/morg/students/sagiahrac/checkpoint-backups/moe1b-two-expert/${RUN_ID}"
RANK_MICROBATCH_SIZE=$((MICROBATCH_SEQUENCES * 2048))
GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SEQUENCES * 2048))

mkdir -p \
    "${LOCAL_ROOT}/data" \
    "${LOCAL_ROOT}/cache/${SLURM_JOB_ID}" \
    "${LOCAL_RUN_DIR}" \
    "${PERSISTENT_ROOT}/runs" \
    "${SUBMIT_DIR}/slurm_output"

(
    flock -x 9
    if [[ ! -f "${LOCAL_SOURCE}/.ready" ]]; then
        mkdir -p "${LOCAL_SOURCE}/src"
        rsync -a --delete "${SOURCE_DIR}/src/" "${LOCAL_SOURCE}/src/"
        touch "${LOCAL_SOURCE}/.ready"
    fi
    if [[ ! -s "${LOCAL_DATA}" ]]; then
        rsync -a "${NFS_DATA}" "${LOCAL_DATA}.partial"
        mv "${LOCAL_DATA}.partial" "${LOCAL_DATA}"
    fi
) 9>"${LOCAL_ROOT}/.stage.lock"

latest_checkpoint="$(
    find "${PERSISTENT_ROOT}/runs" -mindepth 1 -maxdepth 1 -type d -name 'step*' \
        -print 2>/dev/null | sort -V | tail -n 1
)"
if [[ -n "${latest_checkpoint}" && -s "${latest_checkpoint}/.metadata.json" ]]; then
    checkpoint_name="$(basename "${latest_checkpoint}")"
    if [[ ! -s "${LOCAL_RUN_DIR}/${checkpoint_name}/.metadata.json" ]]; then
        rsync -a "${latest_checkpoint}/" "${LOCAL_RUN_DIR}/${checkpoint_name}/"
        echo "restored_checkpoint=${checkpoint_name}"
    fi
fi

export PYTHONPATH="${LOCAL_SOURCE}/src"
export PYTHONUNBUFFERED=1
export OLMO_CORE_USE_TORCH_GROUPED_MM=1
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache/${SLURM_JOB_ID}"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1
export WANDB_MODE=online
export WANDB_RESUME=allow
export WANDB_ENTITY=MoEPaper
export WANDB_PROJECT=MoE
export WANDB_RUN_ID="${RUN_ID}"

echo "host=$(hostname)"
echo "routing=${ROUTING}"
echo "run_id=${RUN_ID}"
echo "architecture=d1024-l9-h8-experts2-top1"
echo "microbatch_sequences=${MICROBATCH_SEQUENCES}"
echo "global_batch_sequences=${GLOBAL_BATCH_SEQUENCES}"
echo "gradient_accumulation_steps=$((GLOBAL_BATCH_SEQUENCES / MICROBATCH_SEQUENCES))"
sha256sum \
    "${LOCAL_SOURCE}/src/olmo_core/nn/moe/router.py" \
    "${LOCAL_SOURCE}/src/scripts/train/moe-1b.py"

replicate_checkpoints() {
    local metadata checkpoint_name destination temporary
    shopt -s nullglob
    for metadata in "${LOCAL_RUN_DIR}"/step*/.metadata.json; do
        checkpoint_name="$(basename "$(dirname "${metadata}")")"
        destination="${PERSISTENT_ROOT}/runs/${checkpoint_name}"
        if [[ -s "${destination}/.metadata.json" ]]; then
            continue
        fi
        temporary="${PERSISTENT_ROOT}/runs/.${checkpoint_name}.${SLURM_JOB_ID}.copying"
        if [[ -e "${temporary}" ]]; then
            continue
        fi
        rsync -a "$(dirname "${metadata}")/" "${temporary}/"
        test -s "${temporary}/.metadata.json"
        mv "${temporary}" "${destination}"
        echo "replicated_checkpoint=${checkpoint_name}"
    done
}

(
    while true; do
        replicate_checkpoints
        sleep 60
    done
) &
replicator_pid=$!
trap 'kill "${replicator_pid}" 2>/dev/null || true' EXIT

"${PYTHON}" -u \
    "${LOCAL_SOURCE}/src/scripts/train/moe-1b.py" \
    "${RUN_NAME}" \
    "--routing=${ROUTING}" \
    "--model.block.feed_forward_moe.num_experts=2" \
    "--model.block.feed_forward_moe.router.top_k=1" \
    "--model.block.feed_forward_moe.router.normalize_expert_weights=null" \
    "--dataset.paths=[${LOCAL_DATA}]" \
    "--dataset.work_dir=${LOCAL_ROOT}/dataset-cache" \
    "--data_loader.global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "--data_loader.num_workers=0" \
    "--train_module.rank_microbatch_size=${RANK_MICROBATCH_SIZE}" \
    "--train_module.compile_model=false" \
    "--trainer.save_folder=${LOCAL_RUN_DIR}" \
    "--trainer.metrics_collect_interval=10" \
    "--trainer.callbacks.wandb.group=moe1b-two-expert-forward-loo" \
    "--trainer.callbacks.checkpointer.save_interval=100" \
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=25" \
    "--trainer.callbacks.lm_evaluator.enabled=false" \
    "--trainer.no_evals"

replicate_checkpoints
