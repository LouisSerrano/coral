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

dataset_name="navier-stokes"


python3 training/tf_regression.py "functa.run_name=distinctive-puddle-2410" "functa.dataset_name=navier-stokes-256" "optim.warmup_epochs=500" "transformer.gamma=5" "transformer.embed_dim=192" "transformer.num_heads=6" "transformer.depth=6" "transformer.dropout=0.1" "optim.weight_decay=5e-2" "optim.batch_size=256" "optim.lr=1e-3" "optim.epochs=15000"
