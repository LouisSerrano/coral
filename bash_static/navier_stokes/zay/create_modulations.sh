#!/bin/bash
#SBATCH -A mdw@v100
#SBATPH --partition=gpu_p2
#SBATCH --job-name=siren    # job name
#SBATCH --ntasks=1                   # number of MP tasks
#SBATCH --ntasks-per-node=1          # number of MPI tasks per node
#SBATCH --qos=qos_gpu-t4
#SBATCH --gres=gpu:1                 # number of GPUs per node
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --time=100:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=slurm_run/navier_stokes/siren-%j.out # output file name
#SBATCH --error=slurm_run/navier_stokes/siren-%j.err  # error file name

set -x
cd ${SLURM_SUBMIT_DIR}

module purge
module load pytorch-gpu/py3/1.10.1 # pytorch-gpu/py3/1.5.0

dataset_name="navier-stokes"
run_name="fine-snowflake-15"

srun python3 -m training.create_modulations "functa.dataset_name=${dataset_name}" "functa.run_name=${run_name}"
