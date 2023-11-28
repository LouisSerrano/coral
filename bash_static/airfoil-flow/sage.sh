#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --job-name=sage

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/%x-%j.out

#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral


python3 template/static/baseline.py --config-name=sage.yaml data.dataset_name=airfoil-flow 
