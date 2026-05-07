#!/bin/bash
#SBATCH --job-name=check-alignment
#SBATCH --output=slurm_output/check-alignment-%j.out
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
python check_alignment.py
