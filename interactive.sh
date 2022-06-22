#!/bin/bash 

LOGDIR=${SCRATCH}/ML_Hydro_train/logs
mkdir -p ${LOGDIR}
args="${@}"

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u --ntasks-per-node 4 --cpus-per-task 32 --gpus-per-node 4 shifter -V ${LOGDIR}:/logs --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
