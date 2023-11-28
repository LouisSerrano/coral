#!/bin/bash
#SBATCH --partition=electronic
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
sub_tr=1
w0=5
model_type="bacon"
base_freq_multiplier=1
quantization_multiplier=2
hidden_dim=64
latent_dim=64
hypernet_depth=1
hypernet_width=256
lr_inr=5e-4
lr_modulations=5e-4
lr_code=0.01
gamma_step=0.9
number_of_lr_steps=20
batch_size=32
epochs=12000
lr_mlp=5e-4
final_lr_value=1e-6
weight_decay_mlp=1e-2
use_scheduler=True
resnet=False
activation=swish
mlp_iters_per_epoch=1
mlp_width=256
mlp_depth=4
batch_size_mlp=64

python3 training/different_inr_regression.py "data.dataset_name=${dataset_name}" "data.sub_tr=${sub_tr}"  "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "inr.hypernet_depth=${hypernet_depth}" "inr.hypernet_width=${hypernet_width}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "inr.model_type=${model_type}" "inr.w0=${w0}" "inr.base_freq_multiplier=${base_freq_multiplier}" "inr.quantization_multiplier=${quantization_multiplier}" "optim.lr_inr=${lr_inr}"  "optim.lr_code=${lr_code}" "optim.lr_modulations=${lr_modulations}"  "optim.gamma_step=${gamma_step}" "optim.number_of_lr_steps=${number_of_lr_steps}" "inr.input_scales=[1/8, 1/8, 1/4, 1/2]" "inr.output_layers=[2, 3]" "optim_mlp.lr_mlp=${lr_mlp}" "optim_mlp.weight_decay=${weight_decay_mlp}" "optim_mlp.use_scheduler=${use_scheduler}" "optim_mlp.mlp_iters_per_epoch=${mlp_iters_per_epoch}" "mlp.width=${mlp_width}" "mlp.depth=${mlp_depth}" "mlp.resnet=${resnet}" "optim_mlp.final_lr_value=${final_lr_value}" "mlp.activation=${activation}" "optim_mlp.batch_size=${batch_size_mlp}"
