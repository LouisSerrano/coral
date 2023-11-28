#!/bin/bash
#SBATCH --partition=funky
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

dataset_name="burgers"
run_name="copper-glitter-937"
lr_mlp=1e-2
final_lr_value=1e-4
weight_decay=1e-4
resnet=True
depth=4
width=384
dropout=0
activation=swish

srun python3 -m training.code_regression "functa.dataset_name=${dataset_name}" "functa.run_names=[${run_name}]" "optim.lr_mlp=${lr_mlp}" "optim.final_lr_value=${final_lr_value}" "optim.weight_decay=${weight_decay}" "mlp.resnet=${resnet}" "mlp.depth=${depth}" "mlp.width=${width}" "mlp.dropout=${dropout}" "mlp.activation=${activation}"
