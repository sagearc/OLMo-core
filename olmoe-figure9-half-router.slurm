#!/bin/bash
#SBATCH --job-name=fig9-half-router
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --error=slurm_output/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-210
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:amd:1
#SBATCH --time=2-00:00:00

set -euo pipefail

: "${SOURCE_DIR:?Set SOURCE_DIR to an immutable source snapshot}"
: "${ROUTING:?Set ROUTING to deepseek or leave_one_out}"
: "${MICROBATCH_SEQUENCES:=64}"
: "${SMOKE_STEPS:=0}"
: "${RUN_TAG:=fig9-half-router}"

case "$ROUTING" in
    deepseek|leave_one_out) ;;
    *)
        echo "ROUTING must be deepseek or leave_one_out, got: $ROUTING" >&2
        exit 2
        ;;
esac
if [[ ! "$MICROBATCH_SEQUENCES" =~ ^[0-9]+$ ]] || ((MICROBATCH_SEQUENCES < 1)); then
    echo "MICROBATCH_SEQUENCES must be a positive integer" >&2
    exit 2
fi
if ((256 % MICROBATCH_SEQUENCES != 0)); then
    echo "MICROBATCH_SEQUENCES must divide the 256-sequence global batch" >&2
    exit 2
fi

SEQUENCE_LENGTH=2048
RANK_MICROBATCH_SIZE=$((MICROBATCH_SEQUENCES * SEQUENCE_LENGTH))
GRAD_ACCUM_STEPS=$((256 / MICROBATCH_SEQUENCES))
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_RUNTIME=/tmp/sagiahrac-olmo-runtime-rocm72
LOCAL_ROOT="/tmp/sagiahrac-${RUN_TAG}-${ROUTING}-${SLURM_JOB_ID}"
LOCAL_SRC="${LOCAL_ROOT}/source/src"
LOCAL_PYTHON="${BASE_RUNTIME}/python/bin/python3.12"
LOCAL_SITE="${BASE_RUNTIME}/site-packages"
SOURCE_SITE="${SUBMIT_DIR}/.venv/lib/python3.12/site-packages"
RUN_NAME="${RUN_TAG}-${SLURM_JOB_ID}"
PERSISTENT_ROOT="/home/morg/students/sagiahrac/checkpoint-backups/fig9-half-router/${RUN_NAME}-${ROUTING}"

if [[ ! -f "${BASE_RUNTIME}/.ready" ]]; then
    echo "Node-local ROCm runtime is not ready: ${BASE_RUNTIME}" >&2
    exit 3
fi
if [[ ! -f "${SOURCE_DIR}/src/scripts/train/olmoe-figure9-half-router.py" ]]; then
    echo "Invalid source snapshot: ${SOURCE_DIR}" >&2
    exit 3
fi

mkdir -p \
    "${LOCAL_SRC}" \
    "${LOCAL_ROOT}/work" \
    "${LOCAL_ROOT}/cache" \
    "${LOCAL_ROOT}/runs" \
    "${PERSISTENT_ROOT}" \
    "${SUBMIT_DIR}/slurm_output"
rsync -a "${SOURCE_DIR}/src/" "${LOCAL_SRC}/"

export OLMO_CORE_USE_TORCH_GROUPED_MM=1
export PYTHONUNBUFFERED=1
export OLMO_DATA_ROOT=/home/morg/students/sagiahrac/dataset/olmoe-1pct
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/torchinductor"
export CACHED_PATH_CACHE_ROOT="${LOCAL_ROOT}/cached_path"
export PYTHONPATH="${LOCAL_SITE}:${LOCAL_SRC}:${SOURCE_SITE}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

if ((SMOKE_STEPS > 0)); then
    export WANDB_MODE=disabled
else
    : "${WANDB_API_KEY:?Pass the W&B key through the submitted environment}"
    export WANDB_MODE=online
fi

echo "host=$(hostname)"
echo "source_dir=${SOURCE_DIR}"
echo "routing=${ROUTING} run_name=${RUN_NAME}"
echo "microbatch_sequences=${MICROBATCH_SEQUENCES} rank_microbatch_tokens=${RANK_MICROBATCH_SIZE}"
echo "global_batch_tokens=$((256 * SEQUENCE_LENGTH)) grad_accum_steps=${GRAD_ACCUM_STEPS}"

LOCAL_RUN_DIR="${LOCAL_ROOT}/runs/${RUN_NAME}-${ROUTING}"
args=(
    "${RUN_NAME}"
    "--routing=${ROUTING}"
    "--train_module.rank_microbatch_size=${RANK_MICROBATCH_SIZE}"
    "--trainer.save_folder=${LOCAL_RUN_DIR}"
)

if ((SMOKE_STEPS > 0)); then
    args+=(
        "--train_module.compile_model=false"
        "--data_loader.num_workers=0"
        "--trainer.metrics_collect_interval=1"
        "--trainer.callbacks.wandb.enabled=false"
        "--trainer.callbacks.lm_evaluator.enabled=false"
        "--trainer.no_evals"
        "--trainer.no_checkpoints"
        "--trainer.hard_stop={unit: steps, value: ${SMOKE_STEPS}}"
    )
fi

replicate_complete_checkpoints() {
    local metadata checkpoint_name destination temporary_destination
    shopt -s nullglob
    for metadata in "${LOCAL_RUN_DIR}"/step*/.metadata.json; do
        checkpoint_name="$(basename "$(dirname "$metadata")")"
        destination="${PERSISTENT_ROOT}/runs/${checkpoint_name}"
        if [[ -s "${destination}/.metadata.json" ]]; then
            continue
        fi
        temporary_destination="${PERSISTENT_ROOT}/runs/.${checkpoint_name}.${SLURM_JOB_ID}.copying"
        mkdir -p "${PERSISTENT_ROOT}/runs"
        flock -w 600 \
            "/home/morg/students/sagiahrac/checkpoint-backups/fig9-half-router/.copy.lock" \
            rsync -a "$(dirname "$metadata")/" "${temporary_destination}/"
        if [[ ! -s "${temporary_destination}/.metadata.json" ]]; then
            echo "Checkpoint replication validation failed: ${checkpoint_name}" >&2
            return 1
        fi
        mv "${temporary_destination}" "${destination}"
        echo "replicated_checkpoint=${checkpoint_name}"
    done
}

cd "${SUBMIT_DIR}"
set +e
srun --kill-on-bad-exit=1 \
    "${LOCAL_PYTHON}" -u \
    "${LOCAL_SRC}/scripts/train/olmoe-figure9-half-router.py" \
    "${args[@]}" &
trainer_pid=$!

replicator_pid=""
if ((SMOKE_STEPS == 0)); then
    (
        while kill -0 "$trainer_pid" 2>/dev/null; do
            replicate_complete_checkpoints || exit 1
            sleep 60
        done
        replicate_complete_checkpoints
    ) &
    replicator_pid=$!
fi

wait "$trainer_pid"
trainer_status=$?
set -e

if [[ -n "$replicator_pid" ]]; then
    wait "$replicator_pid"
fi
if ((SMOKE_STEPS == 0)); then
    replicate_complete_checkpoints
fi

exit "$trainer_status"
