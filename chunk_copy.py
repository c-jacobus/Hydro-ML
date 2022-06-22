import numpy as np
import h5py
import torch
import torch.nn.functional as nn
from torch.nn.parallel import DistributedDataParallel
from networks import UNet as UNet
from networks import New_UNet as New_UNet
import argparse
import resource
import os
import h5py
from collections import OrderedDict
import datetime
import time
import sys
import logging
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


datapath = '/path/to/normalized/h5'

parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default='/pscratch/sd/c/cjacobus/ml-pm-training-2022/logs/hydro_vanilla/4GPU/00/infer_vanilla_size_128_trim_64.h5', type=str)
parser.add_argument("--size", default=128, type=int)
parser.add_argument("--gpu", default=False, type=bool)
parser.add_argument("--temp_path", default='/pscratch/sd/z/zarija/MLHydro/infer_vanilla_512.hdf5', type=str)
args = parser.parse_args()

size=args.size
trim=0
dtype=np.single

world_rank = 0
local_rank = 0


if args.gpu:
    print("GPU MODE")
    from mpi4py import MPI
    world_size = MPI.COMM_WORLD.Get_size()
    world_rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    local_rank = int(os.environ['SLURM_LOCALID'])




    
if args.gpu:
    hf = h5py.File(args.temp_path, 'a', driver='mpio', comm=MPI.COMM_WORLD)
else:
    hf = h5py.File(args.temp_path, 'a')

print("Rank {} initialized file".format(world_rank))

with h5py.File(args.datapath, 'r') as f:
    
    if  world_rank==0: print("Input data path: {}".format(args.datapath))
    full_dim = f['native_fields']['baryon_density'][0,0,:].shape[0]
    if  world_rank==0: print("Dimension: {}".format(full_dim))
        
    slices=int(full_dim/size)
    out_shape = full_dim
    out_size = size
    
    if  world_rank==0:
        print("Dimension: {}".format(full_dim))
        print("Trimmed chunk size: {}".format(size))
        print("Trim: {}".format(trim))
        print("Slices: {}".format(slices))
        print("Sending {} chunks individually...".format(slices**3))
        
        
    chunk_done = [[[False for x in range(slices)] for y in range(slices)] for z in range(slices)]
    #final=torch.empty((1,5,full_dim,full_dim,full_dim), dtype=torch.float32)

    time.sleep(world_rank*5)
    
    if True: 
        for x in range(slices):
            x1 = int(x*size)
            x2 = int(min((x+1)*size,full_dim))
            
            x1_out = int(x*out_size)
            x2_out = int(min((x+1)*out_size, out_shape))
            for y in range(slices):
                y1 = int(y*size)
                y2 = int(min((y+1)*size,full_dim))
                
                y1_out = int(y*out_size)
                y2_out = int(min((y+1)*out_size, out_shape))
                for z in range(slices):
                    z1 = int(z*size)
                    z2 = int(min((z+1)*size,full_dim))
                    
                    z1_out = int(z*out_size)
                    z2_out = int(min((z+1)*out_size, out_shape))
                    
                    if not hf['native_fields']['baryon_density'][x1_out+2,y1_out+2,z1_out+2] == 100:
                        time.sleep(world_rank*0.5)
                        if not hf['native_fields']['baryon_density'][x1_out+2,y1_out+2,z1_out+2] == 100:
                            hf['native_fields']['baryon_density'][x1_out+2,y1_out+2,z1_out+2] = 100
                            print("Rank {} claimed chunk [{},{},{}]".format(world_rank,x,y,z))

                            # load chunk
                            sliced_in_rho = f['native_fields']['baryon_density'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vx = f['native_fields']['velocity_x'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vy = f['native_fields']['velocity_y'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vz = f['native_fields']['velocity_z'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_temp = f['native_fields']['temperature'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            print("Rank {} received chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in_rho.shape))

                            # write to file   
                            hf['native_fields']['baryon_density'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = sliced_in_rho
                            hf['native_fields']['velocity_x'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = sliced_in_vx
                            hf['native_fields']['velocity_y'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = sliced_in_vy
                            hf['native_fields']['velocity_z'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = sliced_in_vz
                            hf['native_fields']['temperature'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = sliced_in_temp                            
                            print("Rank {} wrote chunk [{},{},{}] to file".format(world_rank,x,y,z))


hf.close()

maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')
