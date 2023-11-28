#!/bin/bash
#SBATCH -A mdw@v100
#SBATPH --partition=gpu_p2
#SBATCH --job-name=siren    # job name
#SBATCH --ntasks=8                   # number of MP tasks
#SBATCH --ntasks-per-node=4          # number of MPI tasks per node
#SBATCH --qos=qos_gpu-t4
#SBATCH --gres=gpu:4                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_run/2D/siren-%j.out # output file name
#SBATCH --error=slurm_run/2D/siren-%j.err  # error file name


set -x
cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.1 # pytorch-gpu/py3/1.5.0

#srun python -m training.navier_stokes.siren_template 'optim.batch_size=8' 'optim.batch_size_val=8' 'inr.latent_dim=512' 'inr.depth=10' 'inr.hidden_dim=1024' 'optim.lr_inr=3e-6' 'optim.epochs=1000' 'data.sub=1' 'inr.w0=50'