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

python3 template/static/design_inr.py "data.dataset_name=pipe" 'optim.batch_size=16' 'optim.epochs=5000' 'inr_in.w0=5' 'inr_out.w0=10' 'inr_in.hidden_dim=128' 'inr_in.depth=5' 'inr_out.hidden_dim=128' 'inr_out.depth=5' 'optim.lr_inr=5e-5' 'optim.meta_lr_code=5e-5' 

