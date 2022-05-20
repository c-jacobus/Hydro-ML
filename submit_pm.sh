#!/bin/bash 
#SBATCH -C gpu
#SBATCH -A m3900_g
#SBATCH --ntasks-per-node 4
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-node 4
#SBATCH --time=0:10:00
#SBATCH --image=romerojosh/containers:sc21_tutorial
#SBATCH -o %x-%j.out

LOGDIR=${SCRATCH}/ml-pm-training-2022/logs
mkdir -p ${LOGDIR}
args="${@}"

hostname

# Copy data to tmp, if using
cp /pscratch/sd/p/pharring/Nyx_nbod2hydro/normalized_data/*.h5 /tmp

# Use the head node of the job as the main communicator
export MASTER_ADDR=$(hostname)

set -x
srun -u shifter -V ${LOGDIR}:/logs \
    bash -c "
    source export_DDP_vars.sh
    python train.py ${args}
    "
