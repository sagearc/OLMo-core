#!/bin/bash
#SBATCH --job-name=moe-1b
#SBATCH --output=/home/morg/students/sagiahrac/repos/olmo-core-repo/slurm_output/moe-1b-%x-%j.out
#SBATCH --error=/home/morg/students/sagiahrac/repos/olmo-core-repo/slurm_output/moe-1b-%x-%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-210
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:amd:1
#SBATCH --time=5-00:00:00

# Required env var: ROUTING (one of: baseline, deepseek, ema, ema_seq_aux)
# Submit with:
#   sbatch --job-name=moe-1b-baseline --export=ROUTING=baseline moe-1b.slurm

if [ -z "$ROUTING" ]; then
    echo "ERROR: ROUTING env var not set" >&2
    exit 1
fi

export ROCM_HOME=/opt/rocm-6.4.3

cd /home/morg/students/sagiahrac/repos/olmo-core-repo

source .venv/bin/activate

torchrun \
    --standalone \
    --nproc_per_node=1 \
    src/scripts/train/moe-1b.py \
    moe-1b-${SLURM_JOB_ID} \
    --routing=${ROUTING}
