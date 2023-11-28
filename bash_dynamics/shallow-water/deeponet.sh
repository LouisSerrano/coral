#!/bin/bash
#SBATCH --partition=funky
#SBATCH --job-name=deeponet
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/deeponet/%x-%j.out
#SBATCH --error=slurm_run/deeponet/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

dataset_name='shallow-water-dino'
same_grid=False
sub_from=2
sub_tr=0.2
sub_te=0.2
setting=extrapolation
seq_inter_len=20
seq_extra_len=20

batch_size=10
learning_rate=0.00001
epochs=10000

model_type="mlp"
trunk_depth=4
branch_depth=4
width=100

python3 baseline/deeponet/train.py "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.setting=$setting" "deeponet.model_type=$model_type" "deeponet.branch_depth=$branch_depth" "deeponet.trunk_depth=$trunk_depth" "deeponet.width=$width" "optim.batch_size=$batch_size" "optim.epochs=$epochs" "optim.learning_rate=$learning_rate" "data.dataset_name=$dataset_name" 
