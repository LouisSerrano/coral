#!/bin/bash
<<<<<<< HEAD
#SBATCH --partition=hard
#SBATCH --job-name=mppde
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err
=======
<<<<<<< HEAD
#SBATCH --partition=funky
#SBATCH --job-name=mppde
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/mppde/%x-%j.out
=======

#SBATCH --partition=hard

#SBATCH --constraint "GPUM48G"

#SBATCH --job-name=mppde

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=10000

#SBATCH --output=slurm_run/mppde/%x-%j.out

>>>>>>> d17b5b64186022ce4c2f28b7159abea92bbb920d
#SBATCH --error=slurm_run/mppde/%x-%j.err
>>>>>>> origin

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
<<<<<<< HEAD
conda activate coral


python3 template/test_mppde.py "data.dataset_name=navier-stokes-dino" 'optim.epochs=20000' 'data.sub_from=4' 'data.sub_tr=1' 'data.sub_te=4' 'data.sequence_length=40' 'optim.batch_size=16' 
=======
<<<<<<< HEAD
conda activate coral

dataset_name='navier-stokes-dino'
same_grid=False
sub_from=4
sub_tr=0.05
sub_te=0.05
seq_inter_len=20
seq_extra_len=20
batch_size=64
hidden_features=64
epochs=10000
lr=0.001
unrolling=1

python3 baseline/mppde/train.py "data.sub_tr=$sub_tr" "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.sub_te=$sub_te" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "model.hidden_features=$hidden_features" "optim.batch_size=$batch_size" "optim.epochs=$epochs"  "data.dataset_name=$dataset_name"
=======
conda activate mppde

sub_from=4
sub_tr=1
sub_te=1
setting=extrapolation
sequence_length_in=20
sequence_length_out=20
same_grid=False

batch_size=16
learning_rate=0.001
epochs=10000
neighbors=8 
time_window=1 
unrolling=1 
lr_decay=0.4 
checkpoint_path=''

model_type='GNN'
hidden_features=128

python3 mppde/baseline_coral/mppde_2d_time.py "data.same_grid=$same_grid" "data.setting=$setting" "data.sequence_length_in=$sequence_length_in" "data.sequence_length_out=$sequence_length_out" "data.sub_from=$sub_from" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.batch_size=$batch_size" "optim.learning_rate=$learning_rate" "optim.epochs=$epochs" "optim.neighbors=$neighbors" "optim.time_window=$time_window" "optim.unrolling=$unrolling" "optim.lr_decay=$lr_decay" "model.model_type=$model_type" "model.hidden_features=$hidden_features" "optim.checkpoint_path=$checkpoint_path"
>>>>>>> d17b5b64186022ce4c2f28b7159abea92bbb920d
>>>>>>> origin
