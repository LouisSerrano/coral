#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=errors-graph

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/expe/errors/%x-%j.out

#SBATCH --error=slurm_run/expe/errors/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
batch_size=1

conda deactivate
conda activate mppde

sub_from=4
sub_tr=0.05
sub_te=0.05
python3 expe/errors/errors_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=0.2
sub_te=0.2
python3 expe/errors/errors_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=1
sub_te=1
python3 expe/errors/errors_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"

sub_from=4
sub_tr=0.05
sub_te=0.05
python3 expe/visualization/visualization_errors.py "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=0.2
sub_te=0.2
python3 expe/visualization/visualization_errors.py "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=1
sub_te=1
python3 expe/visualization/visualization_errors.py "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"