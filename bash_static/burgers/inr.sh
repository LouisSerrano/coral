#!/bin/bash
#SBATCH --partition=jazzy
#SBATCH --job-name=functa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/burgers/%x-%j.out
#SBATCH --error=slurm_run/burgers/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

data_to_encode="both"
dataset_name="burgers"
sub_tr=1
w0=5
model_type="siren"
latent_dim=256
hidden_dim=256
hypernet_depth=1
hypernet_width=256
lr_inr=1e-4
lr_code=0.01
batch_size=64
epochs=15000


python3 training/inr.py "data.dataset_name=burgers" "inr.model_type=siren" "inr.w0=50" "inr.hidden_dim=64" "inr.latent_dim=64" "optim.lr_inr=5e-5" "optim.epochs=15000"
