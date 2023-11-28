#!/bin/bash

#SBATCH --partition=hard

#SBATCH --constraint "GPUM48G"

#SBATCH --job-name=up-samplings-times-all

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=5000

#SBATCH --output=slurm_run/expe/up-samplings-times-all/%x-%j.out

#SBATCH --error=slurm_run/expe/up-samplings-times-all/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
batch_size=1


conda activate coral

sub_from=4
sub_tr=0.05
sub_te=0.05
python3 expe/up_sampling/up_sampling_inr.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=0.2
sub_te=0.2
python3 expe/up_sampling/up_sampling_inr.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=1
sub_te=1
python3 expe/up_sampling/up_sampling_inr.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"


conda deactivate
conda activate mppde

sub_from=4
sub_tr=0.05
sub_te=0.05
python3 expe/up_sampling/up_sampling_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=0.2
sub_te=0.2
python3 expe/up_sampling/up_sampling_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_from=4
sub_tr=1
sub_te=1
python3 expe/up_sampling/up_sampling_graph_louis.py "optim.batch_size=$batch_size" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"


conda deactivate
conda activate fno
setting=extrapolation
sequence_length_in=20
sequence_length_out=20

sub_from=4
sub_tr=0.05
sub_te=0.05
python3 expe/up_sampling/up_sampling_operator.py "optim.batch_size=$batch_size" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_tr=0.2
sub_te=0.2
python3 expe/up_sampling/up_sampling_operator.py "optim.batch_size=$batch_size" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"
sub_tr=1
sub_te=1
python3 expe/up_sampling/up_sampling_operator.py "optim.batch_size=$batch_size" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te"

conda deactivate
conda activate coral

python3 expe/visualization/visualization_time.py