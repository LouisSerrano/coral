#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=functa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/navier_stokes/%x-%j.out
#SBATCH --error=slurm_run/navier_stokes/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate torch_nm

dataset_name="navier-stokes"
run_name="expert-oath-612"

srun python3 -m training.code_regression "functa.dataset_name=${dataset_name}" "functa.run_names=[${run_name}]" "optim.lr_mlp=1e-3" "optim.weight_decay=1e-6" 'optim.use_scheduler=True' 'optim.normalize_output=False' 'mlp.resnet=False' 'optim.epochs=100'
