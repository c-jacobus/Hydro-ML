#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -q regular
#SBATCH -J make_train_data
#SBATCH -o logs/%x-%j.out

args="${@}"

hostname

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u --mpi=pmi2 --ntasks-per-node 4 shifter --module gpu --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 bash -c "python make_train_data.py"

