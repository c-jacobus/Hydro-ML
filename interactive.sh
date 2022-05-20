#!/bin/bash 

LOGDIR=${SCRATCH}/ml-pm-training-2022/logs
mkdir -p ${LOGDIR}
args="${@}"

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u --ntasks-per-node 4 --cpus-per-task 32 --gpus-per-node 4 shifter -V ${LOGDIR}:/logs --image=romerojosh/containers:sc21_tutorial \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
