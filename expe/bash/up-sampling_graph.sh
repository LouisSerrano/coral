#!/bin/bash

#SBATCH --partition=electronic

#SBATCH --job-name=up-sampling-graph

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/expe/up-sampling-graph/%x-%j.out

#SBATCH --error=slurm_run/expe/up-sampling-graph/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate mppde

batch_size=1

sub_from=4
sub_tr=0.05
sub_te=0.05

python3 expe/up_sampling/up_sampling/up_sampling_graph.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"