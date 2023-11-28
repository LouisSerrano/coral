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

python3 template/static/static_inr.py "data.dataset_name=airfoil-flow" 'optim.epochs=5000' 'inr_in.w0=5' 'inr_out.w0=5' 'inr_in.latent_dim=128' 'inr_in.hidden_dim=256' 'inr_out.hidden_dim=256' 'inr_out.latent_dim=128' 'model.width=256' 'optim.batch_size=32' 'optim.meta_lr_code=5e-6'


#python3 template/static/single_inr_regression.py "data.dataset_name=airfoil-flow" 'optim.epochs=500' 'inr_in.w0=30' 'inr_out.w0=30' 'inr_in.latent_dim=64' 'inr_in.hidden_dim=64' 'inr_in.depth=4' 'inr_out.hidden_dim=64' 'inr_out.latent_dim=64' 'inr_out.depth=4' 'model.width=64' 'optim.batch_size=32' 'optim.meta_lr_code=5e-6'
