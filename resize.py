import numpy as np
import h5py
import torch
import torch.nn.functional as nn
from torch.nn.parallel import DistributedDataParallel

import argparse
import resource
import os
from os.path import exists
import h5py
from collections import OrderedDict
import datetime
import time
import sys
import logging

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean, pyramid_gaussian

from scipy import ndimage, misc
from scipy.ndimage import gaussian_filter

import numpy.fft
import math
from mpi4py import MPI
from utils.factor_to_fraction import *

'''
This will resize a given h5 file 
using the pyramid resize scheme for hydro fields 
and subsampling dervived fields
in a chunk-wise manner

for gimlet to run on the result, it is recomended that you make a duplicate of a Nyx simulation and overwrite it's fields
'''

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/pscratch/sd/z/zarija/MLHydro/', type=str) # save folder for output
parser.add_argument("--datapath", default='/pscratch/sd/z/zarija/MLHydro/L80_N4096_z3_s2.hdf5', type=str)
parser.add_argument("--trim", default=8, type=int) # width of "crust" trimmed off the inside faces of chunks after resize 
parser.add_argument("--in_Mpc", default=80, type=int) # input field width in Mpc/h
parser.add_argument("--dummy", default=False, type=bool) # for testing purposes
parser.add_argument("--skip", default=False, type=bool) # for testing purposes
parser.add_argument("--template", default=True, type=bool) # whether or not to overwrite a template file, rather than making a new one
parser.add_argument("--temp_path", default='/pscratch/sd/z/zarija/MLHydro/invar3_pyr05_s2.hdf5', type=str)
parser.add_argument("--derived", default=True, type=bool) # whether or not to resize derived fields
parser.add_argument("--naive", default=False, type=bool) # if true: use traditional skimage.transform.resize

args = parser.parse_args()

dtype=np.single

world_rank = 0
local_rank = 0
world_size = MPI.COMM_WORLD.Get_size()
world_rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
local_rank = int(os.environ['SLURM_LOCALID'])


    
# function to resize individual chunks of the large/HD field.
# note different behavior for derived and native fields
def my_resize(field, final_chunk_out, var, factor, naive):
    
    steps = int(math.log2(factor))
    
    if naive:
        field = resize(field, final_chunk_out, anti_aliasing=True)
        
    else:   
        if var == 'tau' or var == 'HI':
            '''
            shape = field.shape
            shape = tuple(int(ti/8) for ti in shape)
            out = np.zeros(shape)
            for x in [3,4]:
                for y in [3,4]:
                    for y in [3,4]:
                        out += field[x::8, y::8, z::8]
            field = out/8
            '''
            field = field[3::8, 3::8, 3::8]

        else:
            for i in range(steps):
                step_chunk_out = tuple([el*(2**(steps-i-1)) for el in final_chunk_out])
                field = gaussian_filter(field, sigma=1/4, mode='reflect', truncate=12.0)
                #field = field[::2, ::2, ::2]

                field = downscale_local_mean(field, (2,2,2) )
    
    return field
                                


if  world_rank==0: print("Input data path: {}".format(args.datapath))
f = h5py.File(args.datapath, 'r')

in_pix = f['native_fields']['baryon_density'][0,0,:].shape[0]
if  world_rank==0: print("Input Dimension: {} pix, {} Mpc".format(in_pix, args.in_Mpc))

if  world_rank==0: print("Resizing for {} pix / {} Mpc".format(512, 80))
trim = args.trim
out_pix = int(512/80 * args.in_Mpc)
if  world_rank==0: print("Output Dimension: {} pix".format(out_pix))

factor = in_pix/out_pix
if  world_rank==0: print("Resize factor: {}".format(factor))

in_step, out_step = FactorToFraction(factor)
if  world_rank==0: print("Resizing: {} --> {}".format(in_step, out_step))

in_chunk_size = in_step
while in_chunk_size < 320:
    in_chunk_size = in_chunk_size*2
    
slices=int(in_pix/in_chunk_size)
out_chunk_size = out_pix/slices

if  world_rank==0:
        print("Trimmed input chunk size: {}".format(in_chunk_size))
        print("Trimmed output chunk size: {}".format(out_chunk_size))
        print("Trim: {}, {}".format(trim*in_step, trim*out_step))
        print("Slices: {}".format(slices))

    
if args.template:
    write_path = args.temp_path
else: 
    write_path = "{}resize_in_{}_out_{}.h5".format(args.folder,in_pix,out_pix)

file_exists = exists(write_path)

if  world_rank==0:
    print("Write path: {}".format(write_path))
    if file_exists: print("File already exists, resuming...")


# initialize the h5 file we're gonna write the resized fields to
with h5py.File(write_path, 'a', driver='mpio', comm=MPI.COMM_WORLD) as hf: 
    
    if not args.template and not file_exists:
        print("File does not exist, initializing...")
        hf.attrs['format'] = "nyx-lyaf"

        dom = hf.create_group("domain")
        dom.attrs['size'] = [80,80,80]
        dom.attrs['shape'] = [out_pix,out_pix,out_pix]

        uni = hf.create_group("universe")
        uni.attrs['hubble'] = 0.675
        uni.attrs['omega_b'] = 0.0487
        uni.attrs['omega_l'] = 0.69
        uni.attrs['omega_m'] = 0.31
        uni.attrs['redshift'] = 2.999999047990147

        rho = hf.create_dataset("native_fields/baryon_density", (out_pix,out_pix,out_pix), dtype='<f4')
        rho.attrs['units'] = "(mean)"

        vx = hf.create_dataset("native_fields/velocity_x", (out_pix,out_pix,out_pix), dtype='<f4')
        vx.attrs['units'] = "cm/s"

        vy = hf.create_dataset("native_fields/velocity_y", (out_pix,out_pix,out_pix), dtype='<f4')
        vy.attrs['units'] = "cm/s"

        vz = hf.create_dataset("native_fields/velocity_z", (out_pix,out_pix,out_pix), dtype='<f4')
        vz.attrs['units'] = "cm/s"

        temp = hf.create_dataset("native_fields/temperature", (out_pix,out_pix,out_pix), dtype='<f4')
        temp.attrs['units'] = "K"
        
        if args.derived:
            HI = hf.create_dataset("/derived_fields/HI_number_density", (out_pix,out_pix,out_pix), dtype='<f4')
            HI.attrs['units'] = "cm**-3"
            
            treal = hf.create_dataset("/derived_fields/tau_real", (out_pix,out_pix,out_pix), dtype='<f4')
            treal.attrs['units'] = "none"
            
            tred = hf.create_dataset("/derived_fields/tau_red", (out_pix,out_pix,out_pix), dtype='<f4')
            tred.attrs['units'] = "none"
    
    if (not file_exists) or args.template:
        hf.create_dataset("chunk_complete", (slices,slices,slices), dtype='<f4')
    
    print("Rank {} initialized file".format(world_rank))
    
    if  world_rank==0: print("Sending {} chunks individually...".format(slices**3))  
    time.sleep(world_rank*0.2) # take turns
    
    # itterate through chunks
    if not args.dummy: 
        for x in range(slices):
            for y in range(slices):
                for z in range(slices):
                    
                    # there was an issue where multiple gpus would resize the same chunk, next 4 lines is a crude work-around 
                    if hf['chunk_complete'][x,y,z] == 0:
                        time.sleep(world_rank*0.1)
                        if hf['chunk_complete'][x,y,z] == 0:
                            hf['chunk_complete'][x,y,z] = 0.5
                            print("Rank {} claimed chunk [{},{},{}]".format(world_rank,x,y,z))
                            
                            start = time.perf_counter()
                            
                            x1_in = int(x*in_chunk_size)
                            if x>0: x1_in-= trim*in_step
                            x2_in = int(min((x+1)*in_chunk_size,in_pix))
                            if x<slices-1: x2_in+= trim*in_step

                            x1_out = int(x*out_chunk_size)
                            x_out_minus = trim*out_step if (x>0) else 0
                            x2_out = int(min((x+1)*out_chunk_size, out_pix))
                            x_out_plus = trim*out_step if (x<slices-1) else 0
                            
                            y1_in = int(y*in_chunk_size)
                            if y>0: y1_in-= trim*in_step
                            y2_in = int(min((y+1)*in_chunk_size,in_pix))
                            if y<slices-1: y2_in+= trim*in_step

                            y1_out = int(y*out_chunk_size)
                            y_out_minus = trim*out_step if (y>0) else 0
                            y2_out = int(min((y+1)*out_chunk_size, out_pix))
                            y_out_plus = trim*out_step if (y<slices-1) else 0
                            
                            z1_in = int(z*in_chunk_size)
                            if z>0: z1_in-= trim*in_step
                            z2_in = int(min((z+1)*in_chunk_size,in_pix))
                            if z<slices-1: z2_in+= trim*in_step

                            z1_out = int(z*out_chunk_size)
                            z_out_minus = trim*out_step if (z>0) else 0
                            z2_out = int(min((z+1)*out_chunk_size, out_pix))
                            z_out_plus = trim*out_step if (z<slices-1) else 0

                            # load chunk
                            sliced_in_rho = f['native_fields']['baryon_density'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            sliced_in_vx = f['native_fields']['velocity_x'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            sliced_in_vy = f['native_fields']['velocity_y'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            sliced_in_vz = f['native_fields']['velocity_z'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            sliced_in_temp = f['native_fields']['temperature'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            if args.derived:
                                sliced_in_HI = f['derived_fields']['HI_number_density'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                                sliced_in_t1 = f['derived_fields']['tau_real'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                                sliced_in_t2 = f['derived_fields']['tau_red'][x1_in:x2_in, y1_in:y2_in, z1_in:z2_in].astype(dtype)
                            
                            stop = time.perf_counter()
                            print("Rank {} received chunk [{},{},{}], input shape: {}, took {}s".format(world_rank,x,y,z,sliced_in_rho.shape,  round(stop-start, 1)))
                            start = time.perf_counter()
    
                            # downsample chunk
                            this_chunk_out_size = (out_chunk_size+x_out_minus+x_out_plus, 
                                                   out_chunk_size+y_out_minus+y_out_plus,
                                                   out_chunk_size+z_out_minus+z_out_plus)
                            
                            
                            chunk_rho = my_resize(sliced_in_rho, this_chunk_out_size, 'rho', factor, args.naive)
                            chunk_vx = my_resize(sliced_in_vx, this_chunk_out_size, 'vel', factor, args.naive)
                            chunk_vy = my_resize(sliced_in_vy, this_chunk_out_size, 'vel', factor, args.naive)
                            chunk_vz = my_resize(sliced_in_vz, this_chunk_out_size, 'vel', factor, args.naive)
                            chunk_temp = my_resize(sliced_in_temp, this_chunk_out_size, 'temp', factor, args.naive)
                            if args.derived:
                                chunk_HI = my_resize(sliced_in_HI, this_chunk_out_size, 'HI', factor, args.naive)
                                chunk_t1 = my_resize(sliced_in_t1, this_chunk_out_size, 'tau', factor, args.naive)
                                chunk_t2 = my_resize(sliced_in_t2, this_chunk_out_size, 'tau', factor, args.naive)

                            stop = time.perf_counter()
                            print("Rank {} resized chunk [{},{},{}], output shape: {}, took {}s".format(world_rank,x,y,z,chunk_rho.shape, round(stop-start, 1)))
                            start = time.perf_counter()
                            
                            
                            x_out_plus = None if (x_out_plus == 0) else -x_out_plus
                            y_out_plus = None if (y_out_plus == 0) else -y_out_plus
                            z_out_plus = None if (z_out_plus == 0) else -z_out_plus
                            
                            #print("Rank {} writing chunk [{},{},{}], output trims: x[{},{}], y[{},{}], z[{},{}] ".format(world_rank, x,y,z,x_out_minus,x_out_plus,y_out_minus,y_out_plus,z_out_minus,z_out_plus))

                            # write to file   
                            hf['native_fields']['baryon_density'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_rho[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                            hf['native_fields']['velocity_x'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vx[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                            hf['native_fields']['velocity_y'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vy[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                            hf['native_fields']['velocity_z'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_vz[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                            hf['native_fields']['temperature'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_temp[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                            
                            if args.derived:
                                hf['derived_fields']['HI_number_density'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_HI[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                                hf['derived_fields']['tau_real'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_t1[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                                hf['derived_fields']['tau_red'][x1_out:x2_out,y1_out:y2_out,z1_out:z2_out] = chunk_t2[x_out_minus:x_out_plus, y_out_minus:y_out_plus, z_out_minus:z_out_plus]
                                
                            stop = time.perf_counter() 
                            print("Rank {} wrote chunk [{},{},{}] to file, took {}s".format(world_rank,x,y,z, round(stop-start, 1)))
                            hf['chunk_complete'][x,y,z] = 1

if not args.dummy:
    hf.close()

maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')
