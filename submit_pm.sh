#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH -q regular
#SBATCH -J trainMLHydro
#SBATCH -o logs/%x-%j.out

LOGDIR=${SCRATCH}/ML_Hydro_train/logs
mkdir -p ${LOGDIR}
args="${@}"

hostname

# Copy data to tmp, if using
cp /pscratch/sd/c/cjacobus/Nyx_512/*.h5 /tmp

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u shifter -V ${LOGDIR}:/logs --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
