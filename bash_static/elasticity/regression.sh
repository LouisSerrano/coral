#!/bin/bash
#SBATCH --partition=jazzy
#SBATCH --job-name=code
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

python3 template/static/design_regression.py "data.dataset_name=elasticity" "inr.run_name=clone-nerf-herder-4289" 'optim.epochs=10000'
