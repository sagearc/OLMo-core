#!/bin/bash
#SBATCH --job-name=moe1b-2e
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
#SBATCH --time=06:00:00
#SBATCH --requeue

set -uo pipefail

: "${SOURCE_DIR:?Set SOURCE_DIR to the immutable source snapshot}"
: "${ROUTING:?Set ROUTING to deepseek or deepseek_binary_leave_one_out}"
: "${RUN_TAG:?Set the shared stable run tag}"
: "${MICROBATCH_SEQUENCES:=96}"
: "${GLOBAL_BATCH_SEQUENCES:=1152}"
: "${SMOKE_STEPS:=0}"
: "${BIAS_GAMMA:=0.001}"
: "${COMPILE_MODEL:=true}"

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
BASE_RUNTIME=/tmp/sagiahrac-olmo-runtime-rocm72
LOCAL_ROOT="/tmp/sagiahrac-moe1b-2e-${RUN_TAG}-${ROUTING}"
LOCAL_SRC="${LOCAL_ROOT}/source/src"
LOCAL_DATA_ROOT=/tmp/sagiahrac-moe1b-2e-data
LOCAL_DATA="${LOCAL_DATA_ROOT}/wiki-part-0-00001.npy"
NFS_DATA=/home/morg/students/sagiahrac/dataset/olmoe-1pct/tokenized/wiki/part-0-00001.npy
LOCAL_PYTHON="${BASE_RUNTIME}/python/bin/python3.12"
LOCAL_SITE="${BASE_RUNTIME}/site-packages"
SOURCE_SITE="${SUBMIT_DIR}/.venv/lib/python3.12/site-packages"
RUN_NAME="${RUN_TAG}"
RUN_ID="${RUN_TAG}-${ROUTING}"
LOCAL_RUN_DIR="${LOCAL_ROOT}/runs/${RUN_ID}"
PERSISTENT_ROOT="/home/morg/students/sagiahrac/checkpoint-backups/moe1b-two-expert/${RUN_ID}"
RANK_MICROBATCH_SIZE=$((MICROBATCH_SEQUENCES * 2048))
GLOBAL_BATCH_SIZE=$((GLOBAL_BATCH_SEQUENCES * 2048))

test -f "${BASE_RUNTIME}/.ready"
test -f "${SOURCE_DIR}/src/scripts/train/moe-1b.py"
mkdir -p \
    "${LOCAL_SRC}" \
    "${LOCAL_DATA_ROOT}" \
    "${LOCAL_ROOT}/cache" \
    "${LOCAL_RUN_DIR}" \
    "${PERSISTENT_ROOT}/runs" \
    "${SUBMIT_DIR}/slurm_output"
rsync -a --delete "${SOURCE_DIR}/src/" "${LOCAL_SRC}/"

(
    flock -x 9
    if [[ ! -s "${LOCAL_DATA}" ]]; then
        rsync -a "${NFS_DATA}" "${LOCAL_DATA}.partial"
        mv "${LOCAL_DATA}.partial" "${LOCAL_DATA}"
    fi
) 9>"${LOCAL_DATA_ROOT}/.copy.lock"

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

export OLMO_CORE_USE_TORCH_GROUPED_MM=1
export PYTHONUNBUFFERED=1
export OLMO_CORE_REPO_DIR="${LOCAL_ROOT}/source"
export OLMO_DATA_ROOT=/home/morg/students/sagiahrac/dataset/olmoe-1pct
export XDG_CACHE_HOME="${LOCAL_ROOT}/cache"
export TRITON_CACHE_DIR="${LOCAL_ROOT}/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${LOCAL_ROOT}/cache/torchinductor"
export CACHED_PATH_CACHE_ROOT="${LOCAL_ROOT}/cache/cached_path"
export PYTHONPATH="${LOCAL_SITE}:${LOCAL_SRC}:${SOURCE_SITE}"
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export RANK=0
export WORLD_SIZE=1
export LOCAL_RANK=0
export LOCAL_WORLD_SIZE=1

wandb_args=()
if ((SMOKE_STEPS > 0)); then
    export WANDB_MODE=disabled
    wandb_args+=("--trainer.callbacks.wandb.enabled=false")
else
    : "${WANDB_API_KEY:?WANDB_API_KEY is required for the long run}"
    export WANDB_MODE=online
    export WANDB_RESUME=allow
    export WANDB_ENTITY=MoEPaper
    export WANDB_PROJECT=MoE
    export WANDB_RUN_ID="${RUN_ID}"
fi

hard_stop_args=()
if ((SMOKE_STEPS > 0)); then
    hard_stop_args+=("--trainer.hard_stop={unit: steps, value: ${SMOKE_STEPS}}")
fi

echo "host=$(hostname)"
echo "routing=${ROUTING}"
echo "run_id=${RUN_ID}"
echo "architecture=d1024-l9-h8-experts2-top1"
echo "microbatch_sequences=${MICROBATCH_SEQUENCES}"
echo "global_batch_sequences=${GLOBAL_BATCH_SEQUENCES}"
echo "gradient_accumulation_steps=$((GLOBAL_BATCH_SEQUENCES / MICROBATCH_SEQUENCES))"
echo "bias_gamma=${BIAS_GAMMA}"
echo "compile_model=${COMPILE_MODEL}"
sha256sum \
    "${LOCAL_SRC}/olmo_core/nn/moe/router.py" \
    "${LOCAL_SRC}/scripts/train/moe-1b.py"

replicate_checkpoints() {
    local metadata checkpoint_name destination temporary
    shopt -s nullglob
    for metadata in "${LOCAL_RUN_DIR}"/step*/.metadata.json; do
        checkpoint_name="$(basename "$(dirname "${metadata}")")"
        # Step 0 is reproducible and copying four multi-GB optimizer checkpoints
        # concurrently only adds avoidable NFS pressure.
        if [[ "${checkpoint_name}" == "step0" ]]; then
            continue
        fi
        destination="${PERSISTENT_ROOT}/runs/${checkpoint_name}"
        if [[ -s "${destination}/.metadata.json" ]]; then
            continue
        fi
        (
            # Serialize large NFS writes from all runs on this node.
            flock -x 9
            if [[ -s "${destination}/.metadata.json" ]]; then
                exit 0
            fi
            temporary="${PERSISTENT_ROOT}/runs/.${checkpoint_name}.${SLURM_JOB_ID}.copying"
            mkdir -p "${temporary}"
            rsync -a "$(dirname "${metadata}")/" "${temporary}/"
            test -s "${temporary}/.metadata.json"
            mv "${temporary}" "${destination}"
            echo "replicated_checkpoint=${checkpoint_name}"
        ) 9>/tmp/sagiahrac-moe1b-two-expert-checkpoint-copy.lock
    done
}

args=(
    "${RUN_NAME}"
    "--routing=${ROUTING}"
    "--model.block.feed_forward_moe.num_experts=2"
    "--model.block.feed_forward_moe.router.top_k=1"
    "--model.block.feed_forward_moe.router.normalize_expert_weights=null"
    "--model.block.feed_forward_moe.router.bias_gamma=${BIAS_GAMMA}"
    "--dataset.paths=[${LOCAL_DATA}]"
    "--dataset.work_dir=${LOCAL_ROOT}/dataset-cache"
    "--data_loader.global_batch_size=${GLOBAL_BATCH_SIZE}"
    "--data_loader.num_workers=0"
    "--train_module.rank_microbatch_size=${RANK_MICROBATCH_SIZE}"
    "--train_module.compile_model=${COMPILE_MODEL}"
    "--trainer.save_folder=${LOCAL_RUN_DIR}"
    "--trainer.metrics_collect_interval=10"
    "--trainer.callbacks.wandb.group=moe1b-two-expert-forward-loo"
    "--trainer.callbacks.checkpointer.save_interval=100"
    "--trainer.callbacks.checkpointer.ephemeral_save_interval=25"
    "--trainer.callbacks.lm_evaluator.enabled=false"
    "--trainer.no_evals"
    "${wandb_args[@]}"
    "${hard_stop_args[@]}"
)

replicator_pid=""
if ((SMOKE_STEPS == 0)); then
    (
        while true; do
            replicate_checkpoints
            sleep 60
        done
    ) &
    replicator_pid=$!
fi

cd "${SUBMIT_DIR}"
"${LOCAL_PYTHON}" -u "${LOCAL_SRC}/scripts/train/moe-1b.py" "${args[@]}"
status=$?

if [[ -n "${replicator_pid}" ]]; then
    kill "${replicator_pid}" 2>/dev/null || true
    wait "${replicator_pid}" 2>/dev/null || true
    replicate_checkpoints
fi
exit "${status}"
