#!/bin/bash
#SBATCH --partition=jazzy
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
dataset_name="burgers"
sub_tr=0.25
sub_te=1
w0=50
model_type="siren"
base_freq_multiplier=0.5
quantization_multiplier=2
inner_steps=3
depth=5
hidden_dim=128
latent_dim=128
lr_inr=3e-6
lr_code=0.01
meta_lr_code=5e-4
weight_decay_code=1e-5
gamma_step=1.0
number_of_lr_steps=20
batch_size=64
epochs=15000
mlp_epochs=10000
lr_mlp=1e-3
final_lr_value=1e-5
weight_decay_mlp=1e-8
use_scheduler=False
activation=swish
mlp_iters_per_epoch=1
mlp_width=512
mlp_depth=4
batch_size_mlp=64
step_size=0.2
inference_model=resnet


python3 training/inr.py "data.dataset_name=${dataset_name}" "data.sub_tr=${sub_tr}" "data.sub_te=${sub_te}" "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "inr.model_type=${model_type}" "inr.w0=${w0}" "inr.base_freq_multiplier=${base_freq_multiplier}" "inr.quantization_multiplier=${quantization_multiplier}" "optim.lr_inr=${lr_inr}"  "optim.lr_code=${lr_code}" "optim.gamma_step=${gamma_step}" "optim.number_of_lr_steps=${number_of_lr_steps}" "inr.input_scales=[1/2, 1/2]" "inr.output_layers=[1]" "optim_mlp.lr_mlp=${lr_mlp}" "optim_mlp.weight_decay=${weight_decay_mlp}" "optim_mlp.use_scheduler=${use_scheduler}" "optim_mlp.mlp_iters_per_epoch=${mlp_iters_per_epoch}" "mlp.width=${mlp_width}" "mlp.depth=${mlp_depth}" "optim_mlp.final_lr_value=${final_lr_value}" "mlp.activation=${activation}" "optim_mlp.batch_size=${batch_size_mlp}" "inr.modulate_scale=False" "inr.modulate_shift=True" "inr.filter_type=gabor" "optim.inner_steps=${inner_steps}" "inr.depth=${depth}" "mlp.inference_model=${inference_model}" "optim.meta_lr_code=${meta_lr_code}" "optim.weight_decay_code=${weight_decay_code}"
