#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=mppde-louis

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=10000

#SBATCH --output=slurm_run/mppde/%x-%j.out

#SBATCH --error=slurm_run/mppde/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate mppde

sub_from=4
sub_tr=0.05
sub_te=0.05
setting=extrapolation
sequence_length_in=20
sequence_length_out=20

batch_size=48
learning_rate=0.001
epochs=25000
neighbors=8 
time_window=1 
unrolling=1 
lr_decay=0.4 
checkpoint_path=''
weight_decay=1e-8

model_type='GNN'
hidden_features=128

python3 mppde/baseline_coral/mppde_2d_time_louis.py "data.same_grid=$same_grid" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.batch_size=$batch_size" "optim.learning_rate=$learning_rate" "optim.epochs=$epochs" "optim.neighbors=$neighbors" "optim.time_window=$time_window" "optim.unrolling=$unrolling" "optim.weight_decay=$weight_decay" "optim.lr_decay=$lr_decay" "model.model_type=$model_type" "model.hidden_features=$hidden_features" "optim.checkpoint_path=$checkpoint_path"