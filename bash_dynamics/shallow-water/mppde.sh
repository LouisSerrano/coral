#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=mppde
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/mppde/%x-%j.out
#SBATCH --error=slurm_run/mppde/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

dataset_name='shallow-water-dino'
#data_to_encode='height'
same_grid=False
sub_from=2
sub_tr=0.05
sub_te=0.05
seq_inter_len=20
seq_extra_len=20
batch_size=16
hidden_features=64
epochs=10000
lr=0.001
unrolling=1

python3 mppde/baseline/train.py "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr=$lr" "optim.unrolling=$unrolling" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "model.hidden_features=$hidden_features" "optim.batch_size=$batch_size" "optim.epochs=$epochs"  "data.dataset_name=$dataset_name" #"data.data_to_encode=$data_to_encode"
