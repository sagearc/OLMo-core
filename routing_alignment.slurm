#!/bin/bash
#SBATCH --job-name=routing-alignment
#SBATCH --output=slurm_output/routing-alignment-%j.out
#SBATCH --error=slurm_output/routing-alignment-%j.err
#SBATCH --partition=gpu
#SBATCH --account=research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --time=00:30:00

export ROCM_HOME=/opt/rocm-6.4.3

REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO_DIR"

source .venv/bin/activate

python analyze_routing_alignment.py \
    --runs moe-1b-283608-ema_centroid moe-1b-284218-baseline_no_loss \
    --n-batches 8 --batch-size 4 \
    --out routing_alignment.png
