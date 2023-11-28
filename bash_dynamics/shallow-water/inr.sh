#!/bin/bash
#SBATCH --job-name=inr
#SBATCH --partition=funky
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=10000
#SBATCH --output=slurm_run/%x-%j.out
#SBATCH --error=slurm_run/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate coral

dataset_name='shallow-water-dino'
data_to_encode='height'

same_grid=False
sub_from=2
sub_tr=0.2
sub_te=0.2
seq_inter_len=20
seq_extra_len=20
batch_size=16
lr_inr=0.000005
epochs=10000
latent_dim=256
depth=6
hidden_dim=256
w0=10
saved_checkpoint=False
#name='expert-galaxy-3879'
#id='cbg99z9r'
#dir='/home/kassai/wandb_logs/wandb/run-20230422_012834-cbg99z9r/files'
#checkpoint_path='/home/kassai/wandb_logs/shallow-water-dino/height/inr/expert-galaxy-3879.pt'

python3 inr/inr.py "data.sub_from=$sub_from" "data.same_grid=$same_grid" "data.dataset_name=$dataset_name" "data.data_to_encode=$data_to_encode" "data.seq_inter_len=$seq_inter_len" "data.seq_extra_len=$seq_extra_len" "data.sub_tr=$sub_tr" "data.sub_te=$sub_te" "optim.lr_inr=$lr_inr" "inr.depth=$depth" "inr.latent_dim=$latent_dim" "inr.hidden_dim=$hidden_dim" "optim.batch_size=$batch_size" "inr.w0=$w0" "optim.epochs=$epochs" "wandb.saved_checkpoint=$saved_checkpoint" #"wandb.name=$name" "wandb.id=$id" "wandb.dir=$dir" "wandb.checkpoint_path=$checkpoint_path"
