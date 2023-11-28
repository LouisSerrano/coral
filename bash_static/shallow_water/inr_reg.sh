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

python3 training/inr_regression.py 'data.dataset_name=shallow-water' 'data.data_to_encode=vorticity' 'optim.batch_size=16' 'optim.batch_size_val=16' 'inr.w0=30' 'inr.hidden_dim=256' 'inr.latent_dim=256' 'inr.depth=6' 'data.sub_tr=0.25' 'data.sub_te=2' 'inr.model_type=siren' 'mlp.inference_model=resnet' 'optim.epochs=10000' 'optim.lr_inr=3e-6' 


#python3 training/inr_regression.py 'data.dataset_name=shallow-water' 'data.data_to_encode=vorticity' 'optim.batch_size=16' 'optim.batch_size_val=16' 'inr.hidden_dim=256' 'inr.latent_dim=256' 'data.sub_tr=2' 'data.sub_te=2' 'inr.model_type=bacon' 'mlp.inference_model=resnet' 'optim.epochs=10000' 'optim.lr_inr=1e-3' 'inr.base_freq_multiplier=1' 'inr.input_scales=[1/8, 1/8, 1/4, 1/2]' 'inr.output_layers=[3]' 'inr.filter_type=gabor' 'inr.modulate_scale=True'


#python3 training/inr_regression.py 'data.dataset_name=shallow-water' 'data.data_to_encode=height' 'optim.batch_size=16' 'optim.batch_size_val=16' 'inr.hidden_dim=256' 'inr.latent_dim=256' 'inr.depth=6' 'data.sub_tr=0.25' 'data.sub_te=2' 'inr.model_type=siren' 'mlp.inference_model=resnet' 'optim.epochs=10000' 'optim.lr_inr=1e-3' 'mlp.width=512' 

