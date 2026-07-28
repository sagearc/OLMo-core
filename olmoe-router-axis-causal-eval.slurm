#!/usr/bin/env bash
#SBATCH --job-name=olmoe-router-axis-eval
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=t-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=logs/router-axis-causal-%j.out
#SBATCH --error=logs/router-axis-causal-%j.err

# Override partition/node at submission only when an explicit fallback is needed.
set -euo pipefail

ACTIVE_REPO="${ACTIVE_REPO:-/home/morg/students/sagiahrac/repos/olmo-core-repo}"
CLEAN_WORKTREE="${CLEAN_WORKTREE:-/home/morg/students/sagiahrac/repos/olmo-core-router-axis-eval}"
SOURCE_COMMIT="${SOURCE_COMMIT:?set SOURCE_COMMIT to the exact pushed commit}"
PHASE="${PHASE:-smoke}"
PERSISTENT_OUTPUT_ROOT="${PERSISTENT_OUTPUT_ROOT:-${ACTIVE_REPO}/output/router-axis-causal-eval}"
DATASET_SPECS_FILE="${DATASET_SPECS_FILE:-${ACTIVE_REPO}/router-axis-validation-datasets.txt}"
DATASET_ROOT="${DATASET_ROOT:-/home/morg/students/sagiahrac/dataset/olmoe-1pct/eval-data/perplexity/v3_small_gptneox20b}"
CUDA_PYTHON="${CUDA_PYTHON:-${ACTIVE_REPO}/.venv-cuda/bin/python}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-8}"
POWER_ITERATIONS="${POWER_ITERATIONS:-12}"

if [[ "$(git -C "${CLEAN_WORKTREE}" rev-parse HEAD)" != "${SOURCE_COMMIT}" ]]; then
  echo "clean worktree is not pinned to SOURCE_COMMIT" >&2
  exit 1
fi
if [[ ! -x "${CUDA_PYTHON}" ]]; then
  echo "missing CUDA Python: ${CUDA_PYTHON}" >&2
  exit 1
fi
JOB_LOCAL_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/router-axis-eval-${SLURM_JOB_ID}}"
JOB_SOURCE="${JOB_LOCAL_ROOT}/source"
# NODE_HF_ROOT may point at an isolated cache from a failed job on the same pinned node.
JOB_HF_ROOT="${NODE_HF_ROOT:-${JOB_LOCAL_ROOT}/hf}"
mkdir -p "${JOB_SOURCE}" "${JOB_HF_ROOT}" "${PERSISTENT_OUTPUT_ROOT}" logs
cp -a "${CLEAN_WORKTREE}/." "${JOB_SOURCE}/"
export PYTHONPATH="${JOB_SOURCE}/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=1

dataset_args=()
if [[ -r "${DATASET_SPECS_FILE}" ]]; then
  while IFS= read -r spec; do
    [[ -z "${spec}" || "${spec}" == \#* ]] && continue
    dataset_args+=(--dataset-spec "${spec}")
  done < "${DATASET_SPECS_FILE}"
else
  while IFS= read -r path; do
    label="$(basename "$(dirname "$(dirname "${path}")")")"
    dataset_args+=(--dataset-spec "${label}=${path}")
  done < <(find "${DATASET_ROOT}" -mindepth 3 -maxdepth 3 -type f -name '*.npy' | sort)
fi
if [[ "${#dataset_args[@]}" -eq 0 ]]; then
  echo "no validation datasets resolved" >&2
  exit 1
fi

driver="${JOB_SOURCE}/src/scripts/olmoe_router_axis_causal_eval.py"

run_revision() {
  local revision="$1"
  local layers="$2"
  local condition_set="$3"
  local sequence_length="$4"
  local calibration_count="$5"
  local eval_count="$6"
  local safe_revision="${revision//[:\/]/-}"
  export HF_HOME="${JOB_HF_ROOT}/${safe_revision}"
  "${CUDA_PYTHON}" "${driver}" \
    --mode run \
    --revision "${revision}" \
    --layers "${layers}" \
    --condition-set "${condition_set}" \
    --device cuda \
    --output-dir "${PERSISTENT_OUTPUT_ROOT}" \
    --sequence-length "${sequence_length}" \
    --calibration-sequences-per-label "${calibration_count}" \
    --eval-sequences-per-label "${eval_count}" \
    --candidate-count "${CANDIDATE_COUNT}" \
    --power-iterations "${POWER_ITERATIONS}" \
    "${dataset_args[@]}"
  if [[ "${HF_HOME}" == "${JOB_HF_ROOT}/"* ]]; then
    rm -rf "${HF_HOME}"
  fi
}

case "${PHASE}" in
  single)
    REVISION="${REVISION:?set REVISION for PHASE=single}"
    LAYERS="${LAYERS:-8}"
    CONDITION_SET="${CONDITION_SET:-pilot}"
    SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-512}"
    CALIBRATION_COUNT="${CALIBRATION_COUNT:-4}"
    EVAL_COUNT="${EVAL_COUNT:-16}"
    run_revision \
      "${REVISION}" \
      "${LAYERS}" \
      "${CONDITION_SET}" \
      "${SEQUENCE_LENGTH}" \
      "${CALIBRATION_COUNT}" \
      "${EVAL_COUNT}"
    ;;
  smoke)
    run_revision "step:5000" "8" "smoke" "256" "1" "1"
    ;;
  pilot)
    for step in 5000 10000 25000 50000 100000 250000 500000 1000000 1220000; do
      run_revision "step:${step}" "0,5,8,10,15" "pilot" "512" "4" "16"
    done
    set +e
    "${CUDA_PYTHON}" "${driver}" \
      --mode gate \
      --output-dir "${PERSISTENT_OUTPUT_ROOT}" \
      --gate-layer 8
    gate_status=$?
    set -e
    if [[ "${gate_status}" -ne 0 ]]; then
      echo "pilot gate did not pass; full sweep will not be launched"
      "${CUDA_PYTHON}" "${driver}" --mode summarize --output-dir "${PERSISTENT_OUTPUT_ROOT}"
      exit 0
    fi
    if [[ "${AUTO_FULL:-1}" != "1" ]]; then
      echo "pilot gate passed; AUTO_FULL is disabled"
      exit 0
    fi
    mapfile -t revisions < <(
      "${CUDA_PYTHON}" "${driver}" --mode list-revisions
    )
    for revision in "${revisions[@]}"; do
      run_revision "${revision}" "8" "full" "512" "4" "16"
    done
    ;;
  full)
    mapfile -t revisions < <(
      "${CUDA_PYTHON}" "${driver}" --mode list-revisions
    )
    for revision in "${revisions[@]}"; do
      run_revision "${revision}" "8" "full" "512" "4" "16"
    done
    ;;
  final)
    run_revision "step:1220000" "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" "final" "512" "4" "16"
    run_revision "step:1220000" "8" "final" "1024" "4" "32"
    ;;
  *)
    echo "unknown PHASE=${PHASE}" >&2
    exit 1
    ;;
esac

"${CUDA_PYTHON}" "${driver}" --mode summarize --output-dir "${PERSISTENT_OUTPUT_ROOT}"
