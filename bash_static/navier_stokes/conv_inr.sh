#!/bin/bash
#SBATCH --partition=electronic
#SBATCH --job-name=functa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/burgers/%x-%j.out
#SBATCH --error=slurm_run/burgers/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate torch_nm


python3 training/conv_inr.py "data.dataset_name=navier-stokes-256" "data.sub_tr=2" "data.sub_te=2"  "inr.model_type=siren" "inr.w0=30" "inr.latent_dim=8" "inr.kernel_dim=8" "inr.depth=4" "inr.hidden_dim=64" "optim.batch_size=64" "optim.lr_inr=5e-5" "optim.epochs=10000" "inr.multiscale_coordinates=True" "inr.interpolation=bilinear"
