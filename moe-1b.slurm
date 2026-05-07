#!/bin/bash
#SBATCH --job-name=moe-1b
#SBATCH --output=slurm_output/moe-1b-%x-%j.out
#SBATCH --error=slurm_output/moe-1b-%x-%j.err
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --time=5-00:00:00

# Required env var: ROUTING (one of: baseline, deepseek, ema, ema_trend).
# ``--export=ALL`` propagates WANDB_API_KEY etc. from the submission env.
# Submit with:
#   sbatch --job-name=moe-1b-baseline --export=ALL,ROUTING=baseline moe-1b.slurm

if [ -z "$ROUTING" ]; then
    echo "ERROR: ROUTING env var not set" >&2
    exit 1
fi
if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY not set in submission env" >&2
    exit 1
fi

# If RUN_NAME is not set, generate one from the job ID (new run).
# To resume an existing run, pass RUN_NAME=moe-1b-<original-job-id>.
RUN_NAME=${RUN_NAME:-moe-1b-${SLURM_JOB_ID}}

export ROCM_HOME=/opt/rocm-6.4.3
export OLMO_CORE_USE_TORCH_GROUPED_MM=1

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"

source .venv/bin/activate

torchrun \
    --standalone \
    --nproc_per_node=1 \
    src/scripts/train/moe-1b.py \
    ${RUN_NAME} \
    --routing=${ROUTING} \
    ${OVERRIDES:-}
