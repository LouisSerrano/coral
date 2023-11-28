#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --job-name=mppde

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/%x-%j.out

#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

sub_tr=4
sub_te=4
data_to_encode=velocity

batch_size=64
lr=0.000005
gamma_step=0.9
lr_code=0.01
meta_lr_code=0
inner_steps=3
test_inner_steps=3
epochs=10000

latent_dim=128
depth=5
hidden_dim=256
w0=10
hypernet_depth=1
hypernet_width=128

python3 template/static/baseline.py --config-name=mppde.yaml 'data.dataset_name=airfoil-flow' 
