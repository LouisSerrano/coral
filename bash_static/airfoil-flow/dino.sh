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

python3 template/static/dino_static.py "data.dataset_name=airfoil-flow" 'inr_in.latent_dim=128' 'optim.epochs=10000' 'model.width=64' 'optim.lr_inr=1e-2'

