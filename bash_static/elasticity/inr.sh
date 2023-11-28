#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=design
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

python3 template/static/design_inr.py "data.dataset_name=elasticity" 'optim.batch_size=64' 'optim.epochs=5000' 'inr_in.w0=10' 'inr_out.w0=15' 'optim.lr_inr=1e-4' 'optim.meta_lr_code=1e-4' 

