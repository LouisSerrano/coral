#!/bin/bash

#SBATCH --partition=jazzy

#SBATCH --job-name=up-sampling-operator

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/expe/up-sampling-operator/%x-%j.out

#SBATCH --error=slurm_run/expe/up-sampling-operator/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate fno

batch_size=1

sub_from=4
sub_tr=1
sub_te=1
setting=extrapolation
sequence_length_in=20
sequence_length_out=20

python3 expe/up_sampling/up_sampling_operator.py "optim.batch_size=$batch_size" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"