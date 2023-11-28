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
dataset_name="navier-stokes"
sub_tr=1
w0=5
model_type="bacon"
base_freq_multiplier=1
quantization_multiplier=1
latent_dim=100
hidden_dim=64
hypernet_depth=1
hypernet_width=256
lr_inr=5e-4
lr_modulations=5e-4
lr_code=0.01
gamma_step=0.9
number_of_lr_steps=20
batch_size=64
batch_size_val=64
epochs=3000

python3 training/inr_meta_sgd.py "data.dataset_name=${dataset_name}" "data.sub_tr=${sub_tr}" "inr.latent_dim=${latent_dim}" "inr.hidden_dim=${hidden_dim}" "inr.hypernet_depth=${hypernet_depth}" "inr.hypernet_width=${hypernet_width}" "optim.epochs=${epochs}" "optim.batch_size=${batch_size}" "optim.batch_size_val=${batch_size_val}" "data.data_to_encode=${data_to_encode}" "inr.model_type=${model_type}" "inr.w0=${w0}" "inr.base_freq_multiplier=${base_freq_multiplier}" "inr.quantization_multiplier=${quantization_multiplier}" "inr.input_scales=[1/8, 1/8, 1/4, 1/2]" "inr.output_layers=[3]" "optim.lr_inr=${lr_inr}"  "optim.lr_code=${lr_code}" "optim.lr_modulations=${lr_modulations}"  "optim.gamma_step=${gamma_step}" "optim.number_of_lr_steps=${number_of_lr_steps}" "inr.modulate_scale=False"
