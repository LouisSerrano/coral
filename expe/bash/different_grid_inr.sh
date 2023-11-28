#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --job-name=different-grid-inr

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/expe/different-grid-inr/%x-%j.out

#SBATCH --error=slurm_run/expe/different-grid-inr/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

batch_size=1
same_grid=False

sub_from=4
sub_tr=0.05
sub_te=0.05

python3 expe/ablation/different_grid.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.same_grid=$same_grid"

sub_from=4
sub_tr=0.2
sub_te=0.2

python3 expe/ablation/different_grid.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.same_grid=$same_grid"
