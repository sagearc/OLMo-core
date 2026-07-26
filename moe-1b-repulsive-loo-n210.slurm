#!/bin/bash
#SBATCH --job-name=moe1b-km-loo-repel-v2
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --error=slurm_output/%x-%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-210
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:amd:1
#SBATCH --time=5-00:00:00

set -euo pipefail

HARD_STOP_STEPS="${HARD_STOP_STEPS:-42386}"
if [[ ! "$HARD_STOP_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "HARD_STOP_STEPS must be a positive integer, got: $HARD_STOP_STEPS" >&2
    exit 2
fi

ROUTING=ema_centroid_leave_one_out
RUN_NAME="moe-1b-${ROUTING}-repulsive-v2-${SLURM_JOB_ID}"
export OLMO_CORE_USE_TORCH_GROUPED_MM=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT="${WANDB_PROJECT:-MoE}"
export OLMO_DISABLE_CHECKPOINTS=0
export OLMO_DATA_ROOT=/home/morg/students/sagiahrac/dataset/olmoe-1pct
export XDG_CACHE_HOME="/tmp/sagiahrac-${ROUTING}-repulsive-v2-${SLURM_JOB_ID}"
export OLMO_SAVE_ROOT="${XDG_CACHE_HOME}/artifacts"
export TRITON_CACHE_DIR="${XDG_CACHE_HOME}/triton"
export TORCHINDUCTOR_CACHE_DIR="${XDG_CACHE_HOME}/torchinductor"
export CACHED_PATH_CACHE_ROOT="${XDG_CACHE_HOME}/cached_path"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

REPO_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
BASE_RUNTIME=/tmp/sagiahrac-olmo-runtime-rocm72
FIXED_SOURCE_ROOT=/tmp/sagiahrac-olmo-source-repulsive-v2
LOCAL_PYTHON="${BASE_RUNTIME}/python/bin/python3.12"
LOCAL_SITE="${BASE_RUNTIME}/site-packages"
SOURCE_SITE="${REPO_DIR}/.venv/lib/python3.12/site-packages"
FIXED_SRC="${FIXED_SOURCE_ROOT}/src"

if [[ ! -f "${BASE_RUNTIME}/.ready" ]]; then
    echo "Node-local ROCm runtime is not ready: ${BASE_RUNTIME}" >&2
    exit 3
fi

# Keep the running experiment isolated from the older workers' imported source.
# The base tree is copied locally, then only the reviewed fix files cross NFS.
if [[ ! -f "${FIXED_SOURCE_ROOT}/.ready" ]]; then
    staging="${FIXED_SOURCE_ROOT}.staging-${SLURM_JOB_ID}"
    mkdir -p "${staging}/src"
    cp -a "${BASE_RUNTIME}/repo/src/." "${staging}/src/"
    cp "${REPO_DIR}/src/olmo_core/nn/moe/router.py" \
        "${staging}/src/olmo_core/nn/moe/router.py"
    cp "${REPO_DIR}/src/scripts/train/moe-1b.py" \
        "${staging}/src/scripts/train/moe-1b.py"
    touch "${staging}/.ready"
    mv "$staging" "$FIXED_SOURCE_ROOT"
fi

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    export WANDB_API_KEY="$(
        /usr/bin/python3 -c \
            'import netrc; auth = netrc.netrc("/home/morg/students/sagiahrac/.netrc").authenticators("api.wandb.ai"); print(auth[2] if auth else "")'
    )"
fi
: "${WANDB_API_KEY:?No W&B API key in the environment or ~/.netrc}"

export OLMO_CORE_REPO_DIR="$REPO_DIR"
export PYTHONPATH="${LOCAL_SITE}:${FIXED_SRC}:${SOURCE_SITE}"
cd "$REPO_DIR"
mkdir -p slurm_output

RUN_ROOT="${OLMO_SAVE_ROOT}/runs/${RUN_NAME}-${ROUTING}"
PERSIST_ROOT="/home/morg/students/sagiahrac/checkpoint-backups/olmoe-repulsive-v2/job${SLURM_JOB_ID}"
mkdir -p "$PERSIST_ROOT"

sync_permanent_checkpoints_once() {
    local checkpoint step destination partial required
    [[ -d "$RUN_ROOT" ]] || return 0
    for checkpoint in "$RUN_ROOT"/step*; do
        [[ -d "$checkpoint" ]] || continue
        step="${checkpoint##*/step}"
        [[ "$step" =~ ^[0-9]+$ ]] || continue
        (( step % 1000 == 0 )) || continue
        for required in \
            "$checkpoint/.metadata.json" \
            "$checkpoint/model_and_optim/.metadata" \
            "$checkpoint/train/rank0.pt"; do
            [[ -s "$required" ]] || continue 2
        done

        destination="${PERSIST_ROOT}/step${step}"
        [[ -d "$destination" ]] && continue
        partial="${destination}.partial-${SLURM_JOB_ID}"
        mkdir -p "$partial"
        rsync -a "$checkpoint/" "$partial/"
        for required in \
            "$partial/.metadata.json" \
            "$partial/model_and_optim/.metadata" \
            "$partial/train/rank0.pt"; do
            [[ -s "$required" ]] || {
                echo "Persistent checkpoint validation failed: $required" >&2
                return 4
            }
        done
        mv "$partial" "$destination"
        echo "Persisted complete checkpoint: $destination"
    done
}

echo "Starting fresh repulsive leave-one-out run on $(hostname)"
echo "Centroid update: selected=no-op, unselected=negative direction, norms=spherical"
echo "Persistent step-1000 checkpoints: $PERSIST_ROOT"

set +e
srun --kill-on-bad-exit=1 \
    "$LOCAL_PYTHON" -u \
    "${FIXED_SRC}/scripts/train/moe-1b.py" \
    "$RUN_NAME" \
    --routing="$ROUTING" \
    --train_module.rank_microbatch_size=$((96 * 2048)) \
    --trainer.hard_stop="{unit: steps, value: ${HARD_STOP_STEPS}}" &
trainer_pid=$!

(
    while kill -0 "$trainer_pid" 2>/dev/null; do
        sync_permanent_checkpoints_once || exit $?
        sleep 60
    done
    sync_permanent_checkpoints_once
) &
backup_pid=$!

wait "$trainer_pid"
trainer_status=$?
wait "$backup_pid"
backup_status=$?
set -e

if (( backup_status != 0 )); then
    echo "Checkpoint backup worker failed with status $backup_status" >&2
    exit "$backup_status"
fi
exit "$trainer_status"
