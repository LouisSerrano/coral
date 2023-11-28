#!/bin/bash
#SBATCH --partition=electronic
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

python3 template/static/static_regression.py "data.dataset_name=airfoil-flow" "inr.run_name=dandy-lion-4438" "optim.epochs=101"
