#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=slurm_output/%x-%j.out
#SBATCH --error=slurm_output/%x-%j.err
#SBATCH --partition=killable
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-23:00:00

# Per-source prep job. Run one of these per source via the driver script.
#
# Required env vars:
#   SOURCE  - one of: dclm, starcoder, pes2o, arxiv, open-web-math, algebraic-stack, wiki
#   DEST    - destination dir (default: $REPO_DIR/dataset/olmoe-100B)
#
# Submit manually:
#   sbatch --job-name=prep-dclm --cpus-per-task=32 \
#       --export=SOURCE=dclm,DEST=/path/to/olmoe-100B \
#       prepare_data.slurm
#
# Or use the driver: ./prepare_data_driver.sh

set -euo pipefail

if [ -z "${SOURCE:-}" ]; then
    echo "ERROR: SOURCE env var not set" >&2
    exit 1
fi

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
DEST="${DEST:-${OLMO_DATA_ROOT:-$REPO_DIR/dataset/olmoe-100B}}"
OLMO_REPO_DIR="${OLMO_REPO_DIR:-$REPO_DIR/../olmo}"
PROCESSES="${SLURM_CPUS_PER_TASK:-8}"

echo "=== prep $SOURCE ==="
echo "  dest:      $DEST"
echo "  processes: $PROCESSES"
echo "  node:      $(hostname)"
echo "  started:   $(date -Iseconds)"

cd "$OLMO_REPO_DIR"
source .venv/bin/activate

python scripts/prepare_olmoe_1pct.py \
    --dest "$DEST" \
    --sources "$SOURCE" \
    --processes "$PROCESSES" \
    --seed 42 \
    --no-parallel

echo "  finished:  $(date -Iseconds)"
