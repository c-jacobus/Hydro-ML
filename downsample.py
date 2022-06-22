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
parser.add_argument("--folder", default='/pscratch/sd/z/zarija/MLHydro/', type=str)
parser.add_argument("--datapath", default='/pscratch/sd/z/zarija/MLHydro/L80_N4096_z3_s1.hdf5', type=str)
parser.add_argument("--trim", default=0, type=int)
parser.add_argument("--in_chunk_size", default=1024, type=int)
parser.add_argument("--out_shape", default=512, type=int)
parser.add_argument("--in_shape", default=4096, type=int)
parser.add_argument("--dummy", default=False, type=bool)
parser.add_argument("--skip", default=False, type=bool)
parser.add_argument("--gpu", default=False, type=bool)
parser.add_argument("--template", default=False, type=bool)
parser.add_argument("--temp_path", default='/pscratch/sd/z/zarija/MLHydro/newer_512.hdf5', type=str)
args = parser.parse_args()

in_chunk_size=args.in_chunk_size
trim=args.trim
out_shape=args.out_shape
in_shape=args.in_shape
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



if not args.dummy:
    
    if args.template:
        if  world_rank==0: print("Writing to: {}".format(args.temp_path))
        if args.gpu:
            hf = h5py.File(args.temp_path, 'a', driver='mpio', comm=MPI.COMM_WORLD)
        else:
            hf = h5py.File(args.temp_path, 'a')
        
    else:  
        if  world_rank==0: print("Writing to: {}down_in_{}_out_{}.h5".format(args.folder,in_shape,out_shape))
        if args.gpu:
            hf = h5py.File("{}down_in_{}_out_{}.h5".format(args.folder,in_shape,out_shape), 'w', driver='mpio', comm=MPI.COMM_WORLD)
        else:
            hf = h5py.File("{}down_in_{}_out_{}.h5".format(args.folder,in_shape,out_shape), 'w')
    
        hf.attrs['format'] = "nyx-lyaf"
        hf.attrs['chunk'] = in_chunk_size
        hf.attrs['trim'] = trim


        dom = hf.create_group("domain")
        dom.attrs['size'] = [80,80,80]
        dom.attrs['shape'] = [out_shape,out_shape,out_shape]

        uni = hf.create_group("universe")
        uni.attrs['hubble'] = 0.675
        uni.attrs['omega_b'] = 0.0487
        uni.attrs['omega_l'] = 0.69
        uni.attrs['omega_m'] = 0.31
        uni.attrs['redshift'] = 2.999999047990147

        rho = hf.create_dataset("native_fields/baryon_density", (out_shape,out_shape,out_shape), dtype='<f4')
        rho.attrs['units'] = "(mean)"

        vx = hf.create_dataset("native_fields/velocity_x", (out_shape,out_shape,out_shape), dtype='<f4')
        vx.attrs['units'] = "cm/s"

        vy = hf.create_dataset("native_fields/velocity_y", (out_shape,out_shape,out_shape), dtype='<f4')
        vy.attrs['units'] = "cm/s"

        vz = hf.create_dataset("native_fields/velocity_z", (out_shape,out_shape,out_shape), dtype='<f4')
        vz.attrs['units'] = "cm/s"

        temp = hf.create_dataset("native_fields/temperature", (out_shape,out_shape,out_shape), dtype='<f4')
        temp.attrs['units'] = "K"

        HI = hf.create_dataset("/derived_fields/HI_number_density", (out_shape,out_shape,out_shape), dtype='<f4')
        HI.attrs['units'] = "cm**-3"
    
    
    
    print("Rank {} initialized file".format(world_rank))

with h5py.File(args.datapath, 'r') as f:
    if not args.dummy: 
        if  world_rank==0: print("Input data path: {}".format(args.datapath))
        in_shape = f['native_fields']['baryon_density'][0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(in_shape))
        

    else:
        # in_shape=1024
        in_shape = f['native_fields']['baryon_density'][0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(in_shape))
        
    
    slices=int(in_shape/in_chunk_size)
    hf.create_dataset("chunk_complete", (slices,slices,slices), dtype='<f4')
    
    out_size = out_shape/slices
    
    if  world_rank==0:
        print("Trimmed chunk size: {}".format(in_chunk_size))
        print("Trim: {}".format(trim))
        print("Slices: {}".format(slices))
        print("Sending {} chunks individually...".format(slices**3))
        
        
    chunk_done = [[[False for x in range(slices)] for y in range(slices)] for z in range(slices)]
    #final=torch.empty((1,5,in_shape,in_shape,in_shape), dtype=torch.float32)

    time.sleep(world_rank*5)
    
    if not args.dummy: 
        for x in range(slices):
            x1 = int(x*in_chunk_size)
            x2 = int(min((x+1)*in_chunk_size,in_shape))
            
            x1_out = int(x*out_size)
            x2_out = int(min((x+1)*out_size, out_shape))
            for y in range(slices):
                y1 = int(y*in_chunk_size)
                y2 = int(min((y+1)*in_chunk_size,in_shape))
                
                y1_out = int(y*out_size)
                y2_out = int(min((y+1)*out_size, out_shape))
                for z in range(slices):
                    z1 = int(z*in_chunk_size)
                    z2 = int(min((z+1)*in_chunk_size,in_shape))
                    
                    z1_out = int(z*out_size)
                    z2_out = int(min((z+1)*out_size, out_shape))
                    
                    if not hf['chunk_complete'][x,y,z] == 1:
                        time.sleep(world_rank*0.5)
                        if not hf['chunk_complete'][x,y,z] == 1:
                            hf['chunk_complete'][x,y,z] = 1
                            print("Rank {} claimed chunk [{},{},{}]".format(world_rank,x,y,z))

                            # load chunk
                            sliced_in_rho = f['native_fields']['baryon_density'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vx = f['native_fields']['velocity_x'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vy = f['native_fields']['velocity_y'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vz = f['native_fields']['velocity_z'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_temp = f['native_fields']['temperature'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            #sliced_in_HI = f['derived_fields']['HI_number_density'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            print("Rank {} received chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in_rho.shape))

                            # downsample chunk
                            chunk_rho = resize(sliced_in_rho, (out_size,out_size,out_size), anti_aliasing=True)
                            chunk_vx = resize(sliced_in_vx, (out_size,out_size,out_size), anti_aliasing=True)
                            chunk_vy = resize(sliced_in_vy, (out_size,out_size,out_size), anti_aliasing=True)
                            chunk_vz = resize(sliced_in_vz, (out_size,out_size,out_size), anti_aliasing=True)
                            chunk_temp = resize(sliced_in_temp, (out_size,out_size,out_size), anti_aliasing=True)
                            #chunk_HI = resize(sliced_in_HI, (out_size,out_size,out_size), anti_aliasing=True)
                            print("Rank {} downsampled chunk [{},{},{}], output shape: {}".format(world_rank,x,y,z,chunk_rho.shape))

                            # write to file   
                            hf['native_fields']['baryon_density'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_rho
                            hf['native_fields']['velocity_x'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vx
                            hf['native_fields']['velocity_y'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vy
                            hf['native_fields']['velocity_z'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vz
                            hf['native_fields']['temperature'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_temp
                            #hf['derived_fields']['HI_number_density'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_HI
                            
                            print("Rank {} wrote chunk [{},{},{}] to file".format(world_rank,x,y,z))

if not args.dummy:
    hf.close()

maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')
