#!/bin/bash
# Submit one prep job per source, with CPU allocation sized to source weight.
# DCLM gets 32 CPUs (longest job, ~4h of dolma tokenization).
# Small sources (wiki, alg-stack, OWM) get 8 CPUs (finish in <30min).
#
# Usage:
#   ./prepare_data_driver.sh                  # submit all 7 sources
#   ./prepare_data_driver.sh wiki arxiv       # submit only these
#   DEST=/custom/path ./prepare_data_driver.sh

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DEST="${DEST:-${OLMO_DATA_ROOT:-$REPO_DIR/dataset/olmoe-100B}}"
SLURM_SCRIPT="${SLURM_SCRIPT:-$REPO_DIR/prepare_data.slurm}"

# Sources in submission order, with CPU allocations sized by file count
# (dolma parallelizes per-file, so extra CPUs beyond file_count are wasted).
# file counts: dclm~23, starcoder~?, pes2o~?, arxiv~47, owm=1, alg-stack=1, wiki=2.
SOURCES_ORDERED=(dclm starcoder pes2o arxiv open-web-math algebraic-stack wiki)
declare -A CPUS=(
    [dclm]=32
    [starcoder]=16
    [pes2o]=8
    [arxiv]=32
    [open-web-math]=2
    [algebraic-stack]=2
    [wiki]=4
)

selected=("$@")
if [ ${#selected[@]} -eq 0 ]; then
    selected=("${SOURCES_ORDERED[@]}")
fi

mkdir -p "$REPO_DIR/slurm_output"

echo "Submitting prep jobs to DEST=$DEST"
for src in "${selected[@]}"; do
    if [ -z "${CPUS[$src]:-}" ]; then
        echo "unknown source: $src" >&2
        exit 1
    fi
    cpus="${CPUS[$src]}"
    jobid=$(sbatch --parsable \
        --job-name="prep-$src" \
        --cpus-per-task="$cpus" \
        --export="SOURCE=$src,DEST=$DEST" \
        "$SLURM_SCRIPT")
    echo "  $src ($cpus CPUs) → job $jobid"
done

echo
echo "Track progress:  squeue --me"
echo "Logs:            tail -f slurm_output/prep-<source>-<jobid>.out"
