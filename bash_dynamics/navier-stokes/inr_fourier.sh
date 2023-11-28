#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=inr
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

python3 template/inr.py --config-name=fourier.yaml 'inr.model_type=fourier_features' 'data.sub_from=4' 'data.sub_tr=0.05' 'data.sub_te=4' 'inr.latent_dim=128' 'inr.hidden_dim=128' 'inr.depth=4' 'optim.batch_size=64' 'optim.meta_lr_code=0' 'optim.lr_inr=5e-4' 'optim.epochs=10000' 'data.sequence_length=2' 'inr.max_frequencies=4' 'inr.num_frequencies=16'
