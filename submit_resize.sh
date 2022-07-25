#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -q regular
#SBATCH -J resize
#SBATCH -o logs/%x-%j.out

LOGDIR=${SCRATCH}/ML_Hydro_train/logs
mkdir -p ${LOGDIR}
args="${@}"

hostname

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u shifter -V ${LOGDIR}:/logs --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python resize.py ${args}
    "
