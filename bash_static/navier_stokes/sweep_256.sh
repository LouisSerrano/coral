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

#wandb agent --count 1 spatiotemp-isir/functa2functa/jjt3rlkz
#wandb agent --count 1 spatiotemp-isir/functa2functa/lqzgynzn
wandb agent --count 1 spatiotemp-isir/functa2functa/rfytwqdl

