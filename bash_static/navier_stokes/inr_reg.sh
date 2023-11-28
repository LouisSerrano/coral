#!/bin/bash
#SBATCH --partition=hard
#SBATCH --job-name=functa
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=5000
#SBATCH --output=slurm_run/burgers/%x-%j.out
#SBATCH --error=slurm_run/burgers/%x-%j.err

source $MINICONDA_PATH/etc/profile.d/conda.sh

set -x
conda init bash
conda activate torch_nm

data_to_encode="both"
dataset_name="navier-stokes-256"
sub_tr=0.05
sub_te=2
w0=20
model_type="siren"
base_freq_multiplier=1
quantization_multiplier=1
hidden_dim=128
latent_dim=128
hypernet_depth=1
hypernet_width=256
lr_inr=3e-5
lr_code=0.01
gamma_step=1
number_of_lr_steps=20
batch_size=32
batch_size_val=32
epochs=15000
lr_mlp=1e-3
weight_decay_code=0
use_scheduler=True
inference_model=resnet
mlp_iters_per_epoch=5
mlp_width=512
mlp_depth=3
mod_activation=None

python3 training/inr_regression.py "data.dataset_name=${dataset_name}" "data.sub_tr=${sub_tr}" "data.sub_te=${sub_te}"   "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "optim.batch_size_val=${batch_size_val}" "inr.model_type=${model_type}" "inr.w0=${w0}" "inr.base_freq_multiplier=${base_freq_multiplier}" "inr.quantization_multiplier=${quantization_multiplier}" "optim.lr_inr=${lr_inr}"  "optim.lr_code=${lr_code}" "optim.gamma_step=${gamma_step}" "optim.number_of_lr_steps=${number_of_lr_steps}" "inr.input_scales=[1/8, 1/8, 1/4, 1/2]" "inr.output_layers=[5]" "optim_mlp.lr_mlp=${lr_mlp}" "optim_mlp.weight_decay=${weight_decay_code}" "optim_mlp.use_scheduler=${use_scheduler}" "optim_mlp.mlp_iters_per_epoch=${mlp_iters_per_epoch}" "mlp.width=${mlp_width}" "mlp.depth=${mlp_depth}" "mlp.inference_model=${inference_model}" "inr.mod_activation=${mod_activation}" 
