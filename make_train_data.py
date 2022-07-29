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

'''
This will normalize fields from two given h5 files 
and write them to a new file in the coarse/fine format used for training  
'''


parser = argparse.ArgumentParser()
parser.add_argument("--datapath", default='/pscratch/sd/z/zarija/MLHydro/L80_N512_z3_s2.hdf5', type=str) # coarse fields
parser.add_argument("--datapath_2", default='/pscratch/sd/z/zarija/MLHydro/invar_pyr05_sub.hdf5', type=str) # fine fields
parser.add_argument("--size", default=128, type=int) # chunk width 

parser.add_argument("--out_shape", default=512, type=int) #output width 
parser.add_argument("--derived", default=False, type=bool) # TRUE: save derived, FALSE: save hydro
parser.add_argument("--flux", default=True, type=bool) # TRUE: save flux, FALSE: save log-normed tau (only if derived=True)
parser.add_argument("--save_path", default='/pscratch/sd/c/cjacobus/Nyx_512/train_s1_512_invar3_hydro.h5', type=str)
args = parser.parse_args()

size=args.size
out_shape=args.out_shape
trim=0
dtype=np.single
file_exists=os.path.exists(args.save_path)

world_rank = 0
local_rank = 0


from mpi4py import MPI
world_size = MPI.COMM_WORLD.Get_size()
world_rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
local_rank = int(os.environ['SLURM_LOCALID'])


f1 = h5py.File(args.datapath, 'r')
f2 = h5py.File(args.datapath_2, 'r')
    
if  world_rank==0: 
    print("Input data path: {}".format(args.datapath))
full_dim = f1['native_fields']['baryon_density'][0,0,:].shape[0]
if  world_rank==0: print("Dimension: {}".format(full_dim))

slices=int(full_dim/size)

if  world_rank==0: 
    if file_exists: print("File already exists, resuming...")     
    
with h5py.File(args.save_path, 'a', driver='mpio', comm=MPI.COMM_WORLD) as hf:

    if not file_exists:
        if  world_rank==0: print("File does not exist, initializing attributes...")
        dom = hf.create_group("domain")
        dom.attrs['size'] = [80,80,80]
        dom.attrs['shape'] = [full_dim,full_dim,full_dim]

        uni = hf.create_group("universe")
        uni.attrs['hubble'] = 0.675
        uni.attrs['omega_b'] = 0.0487
        uni.attrs['omega_l'] = 0.69
        uni.attrs['omega_m'] = 0.31
        uni.attrs['redshift'] = 2.9999991588912964

        # rho = hf.create_dataset("native_fields/baryon_density", data=np.exp(14.*final[0,0,:,:,:]))
        coarse = hf.create_dataset("coarse", (5, full_dim,full_dim,full_dim,), dtype='<f4')
        ch = 1 if args.derived else 5
        fine = hf.create_dataset("fine", (ch, full_dim,full_dim,full_dim), dtype='<f4')
        
        '''
        coarse = hf.create_dataset("coarse", (full_dim,full_dim,full_dim, 5), dtype='<f4')
        fine = hf.create_dataset("fine", (full_dim,full_dim,full_dim, 5), dtype='<f4')
        '''
        
        hf.create_dataset("chunk_complete", (slices,slices,slices), dtype='<f4')

    print("Rank {} initialized file".format(world_rank))


    out_shape = full_dim
    out_size = size

    if  world_rank==0:
        print("Dimension: {}".format(full_dim))
        print("Trimmed chunk size: {}".format(size))
        print("Trim: {}".format(trim))
        print("Slices: {}".format(slices))
        print("Sending {} chunks individually...".format(slices**3))

    time.sleep(world_rank*0.05)


    for x in range(slices):
        x1 = int(x*size)
        x2 = int(min((x+1)*size,full_dim))

        for y in range(slices):
            y1 = int(y*size)
            y2 = int(min((y+1)*size,full_dim))

            for z in range(slices):
                z1 = int(z*size)
                z2 = int(min((z+1)*size,full_dim))

                if not hf['chunk_complete'][x,y,z] == 1:
                    time.sleep(world_rank*0.05)

                    if not hf['chunk_complete'][x,y,z] == 1:
                        hf['chunk_complete'][x,y,z] = 1

                        print("Rank {} claimed chunk [{},{},{}]".format(world_rank,x,y,z))

                        # load chunk
                        sliced_in_rho = f1['native_fields']['baryon_density'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        sliced_in_vx = f1['native_fields']['velocity_x'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        sliced_in_vy = f1['native_fields']['velocity_y'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        sliced_in_vz = f1['native_fields']['velocity_z'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        sliced_in_temp = f1['native_fields']['temperature'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        print("Rank {} received coarse chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in_rho.shape))

                        # normalize
                        sliced_in_rho = np.log(sliced_in_rho)/14
                        sliced_in_vx = sliced_in_vx/9e7
                        sliced_in_vy = sliced_in_vy/9e7
                        sliced_in_vz = sliced_in_vz/9e7
                        sliced_in_temp = np.log(sliced_in_temp)/8 -1.5
                        print("Rank {} normalized coarse chunk [{},{},{}]".format(world_rank,x,y,z))

                        # write to file   
                        hf['coarse'][0, x1:x2, y1:y2, z1:z2] = sliced_in_rho
                        hf['coarse'][1, x1:x2, y1:y2, z1:z2] = sliced_in_vx
                        hf['coarse'][2, x1:x2, y1:y2, z1:z2] = sliced_in_vy
                        hf['coarse'][3, x1:x2, y1:y2, z1:z2] = sliced_in_vz
                        hf['coarse'][4, x1:x2, y1:y2, z1:z2] = sliced_in_temp
                        print("Rank {} wrote coarse chunk [{},{},{}] to file".format(world_rank,x,y,z))

                        # load chunk
                        if not args.derived:
                            sliced_in_rho = f2['native_fields']['baryon_density'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vx = f2['native_fields']['velocity_x'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vy = f2['native_fields']['velocity_y'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_vz = f2['native_fields']['velocity_z'][x1:x2, y1:y2, z1:z2].astype(dtype)
                            sliced_in_temp = f2['native_fields']['temperature'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        else:
                            sliced_in_der = f2['derived_fields']['tau_red'][x1:x2, y1:y2, z1:z2].astype(dtype)
                        print("Rank {} received fine chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in_rho.shape))

                        # normalize
                        if not args.derived:
                            sliced_in_rho = np.log(sliced_in_rho)/14
                            sliced_in_vx = sliced_in_vx/9e7
                            sliced_in_vy = sliced_in_vy/9e7
                            sliced_in_vz = sliced_in_vz/9e7
                            sliced_in_temp = np.log(sliced_in_temp)/8 -1.5
                        else:
                            if args.flux:
                                sliced_in_der = np.exp(-sliced_in_der)
                            else:
                                sliced_in_der = np.log(sliced_in_der+1)/20
                            
                        print("Rank {} normalized fine chunk [{},{},{}]".format(world_rank,x,y,z))

                        # write to file  
                        if not args.derived:
                            hf['fine'][0, x1:x2, y1:y2, z1:z2] = sliced_in_rho
                            hf['fine'][1, x1:x2, y1:y2, z1:z2] = sliced_in_vx
                            hf['fine'][2, x1:x2, y1:y2, z1:z2] = sliced_in_vy
                            hf['fine'][3, x1:x2, y1:y2, z1:z2] = sliced_in_vz
                            hf['fine'][4, x1:x2, y1:y2, z1:z2] = sliced_in_temp
                        else:
                            hf['fine'][0, x1:x2, y1:y2, z1:z2] = sliced_in_der
                        print("Rank {} wrote fine chunk [{},{},{}] to file".format(world_rank,x,y,z))

hf.close()

if  world_rank==0: print("Saved: {}".format(full_dim))
maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')
