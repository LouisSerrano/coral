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

python3 template/static/static_inr.py "data.dataset_name=cylinder-flow" 'optim.epochs=5000' 'inr_in.w0=20' 'inr_out.w0=15' 'inr_in.latent_dim=128' 'inr_out.latent_dim=128' 'optim.epochs=2000' 'model.width=64'

