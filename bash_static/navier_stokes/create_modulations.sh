#!/bin/bash
#SBATCH --partition=electronic
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
run_name="grateful-sweep-1"

srun python3 -m training.create_modulations "functa.dataset_name=${dataset_name}" "functa.run_name=${run_name}"
