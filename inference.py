import numpy as np
import h5py
import torch
import torch.nn.functional as nn
from torch.nn.parallel import DistributedDataParallel
from utils.YParams import YParams
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


datapath = '/path/to/normalized/h5'

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/pscratch/sd/c/cjacobus/ml-pm-training-2022/logs/vanilla/4GPU/00/', type=str)
parser.add_argument("--weights", default='training_checkpoints/ckpt.tar', type=str)
parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
parser.add_argument("--config", default='vanilla', type=str)
parser.add_argument("--datapath", default='/pscratch/sd/p/pharring/Nyx_nbod2hydro/normalized_data/test.h5', type=str)
parser.add_argument("--trim", default=64, type=int)
parser.add_argument("--size", default=512, type=int)
parser.add_argument("--full_dim", default=1024, type=int)
parser.add_argument("--flavor", default='vanilla', type=str)
parser.add_argument("--dummy", default=False, type=bool)
parser.add_argument("--skip", default=False, type=bool)
parser.add_argument("--gpu", default=False, type=bool)
args = parser.parse_args()

params = YParams(os.path.abspath(args.yaml_config), args.config)

size=args.size
trim=args.trim
full_dim=args.full_dim
dtype=np.single
weights_path=os.path.join(args.folder, args.weights)

params.distributed = False
world_rank = 0
local_rank = 0

if(args.gpu):
    from mpi4py import MPI
    world_size = MPI.COMM_WORLD.Get_size()
    world_rank = MPI.COMM_WORLD.Get_rank()
    name = MPI.Get_processor_name()
    local_rank = int(os.environ['SLURM_LOCALID'])

print("World Rank: {}, Local Rank: {}".format(world_rank, local_rank))

if  world_rank==0:
    print("Initializing model...")
    print("Config: {}, {}".format(args.yaml_config, args.config))

if args.flavor == 'next':
    if  world_rank==0: print("Model = New_UNet")
    model = New_UNet.UNet(params)
else:
    if  world_rank==0: print("Model = UNet")
    model = UNet.UNet(params)
    
if  world_rank==0: print("Initialized model [✓]")

if not args.dummy:  
    if  world_rank==0:
        print("Loading checkpoint: {}".format(weights_path))
    
    if(args.gpu):
        checkpoint = torch.load(weights_path, map_location=torch.device('cuda:%d'%local_rank))
    else:
        checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    new_model_state = OrderedDict()
    model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
    for key in checkpoint[model_key].keys():
        if 'module.' in key: # model was stored using ddp which prepends module
            name = str(key[7:])
            new_model_state[name] = checkpoint[model_key][key]
        else:
            new_model_state[key] = checkpoint[model_key][key]

    model.load_state_dict(new_model_state)

else:
    model.apply(model.get_weights_function(params.weight_init))
    
model.eval()

if  world_rank==0:
    if not args.dummy:  
        print("Loaded model checkpoint [✓]")
    else:
        print("Initialized dummy weights [✓]")
        
if not args.dummy:        
    hf = h5py.File("{}inference_{}_size_{}_trim_{}.h5".format(args.folder,args.flavor,size,trim), 'w', driver='mpio', comm=MPI.COMM_WORLD)
    
    hf.attrs['format'] = "nyx-lyaf"
    hf.attrs['chunk'] = size
    hf.attrs['trim'] = trim
    hf.attrs['flavor'] = args.flavor

    dom = hf.create_group("domain")
    dom.attrs['size'] = [21.09375,21.09375,21.09375]
    dom.attrs['shape'] = [1024,1024,1024]

    uni = hf.create_group("universe")
    uni.attrs['hubble'] = 0.675
    uni.attrs['omega_b'] = 0.0487
    uni.attrs['omega_l'] = 0.69
    uni.attrs['omega_m'] = 0.31
    uni.attrs['redshift'] = 2.9999991588912964

    # rho = hf.create_dataset("native_fields/baryon_density", data=np.exp(14.*final[0,0,:,:,:]))
    rho = hf.create_dataset("native_fields/baryon_density", (1024,1024,1024), dtype='<f4')
    rho.attrs['units'] = "(mean)"

    # vx = hf.create_dataset("native_fields/velocity_x", data=final[0,1,:,:,:]*9e7)
    vx = hf.create_dataset("native_fields/velocity_x", (1024,1024,1024), dtype='<f4')
    vx.attrs['units'] = "km/s"

    # vy = hf.create_dataset("native_fields/velocity_y", data=final[0,2,:,:,:]*9e7)
    vy = hf.create_dataset("native_fields/velocity_y", (1024,1024,1024), dtype='<f4')
    vy.attrs['units'] = "km/s"

    # vz = hf.create_dataset("native_fields/velocity_z", data=final[0,3,:,:,:]*9e7)
    vz = hf.create_dataset("native_fields/velocity_z", (1024,1024,1024), dtype='<f4')
    vz.attrs['units'] = "km/s"

    # temp = hf.create_dataset("native_fields/temperature", data=np.exp(8.*(final[0,4,:,:,:] + 1.5)))
    temp = hf.create_dataset("native_fields/temperature", (1024,1024,1024), dtype='<f4')
    temp.attrs['units'] = "K"

    print("Rank {} initialized file".format(world_rank))
    
with h5py.File(args.datapath, 'r') as f:
    if not args.dummy: 
        if  world_rank==0: print("Input data path: {}".format(args.datapath))
        full_dim = f['Nbody'][0,0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(full_dim))
        # full = f['Nbody'][:,:,:,:].astype(dtype)

        # full = np.expand_dims(full, axis=0) # add batch dim
        # full = torch.from_numpy(full)
        # full_dim=full.shape[3]

    else:
        # full_dim=1024
        full_dim = f['Nbody'][0,0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(full_dim))
        
    
    slices=int(full_dim/size)
    
    
    if  world_rank==0:
        print("Dimension: {}".format(full_dim))
        print("Trimmed chunk size: {}".format(size))
        print("Trim: {}".format(trim))
        print("Slices: {}".format(slices))
        print("Sending {} chunks individually...".format(slices**3))
        
        
    chunk_done = [[[False for x in range(slices)] for y in range(slices)] for z in range(slices)]
    #final=torch.empty((1,5,full_dim,full_dim,full_dim), dtype=torch.float32)

    time.sleep(world_rank*5)
    
    if not args.dummy: 
        for x in range(slices):
            x1 = int(x*size)
            x1_edge = trim if (x>0) else 0
            x2 = int(min((x+1)*size,full_dim))
            x2_edge = trim if (x<slices-1) else 0
            for y in range(slices):
                y1 = int(y*size)
                y1_edge = trim if (y>0) else 0
                y2 = int(min((y+1)*size,full_dim))
                y2_edge = trim if (y<slices-1) else 0
                for z in range(slices):
                    z1 = int(z*size)
                    z1_edge = trim if (z>0) else 0
                    z2 = int(min((z+1)*size,full_dim))
                    z2_edge = trim if (z<slices-1) else 0
                    
                    if hf['native_fields']['baryon_density'][x1+2,y1+2,z1+2] == 0:
                        #logging.info("Rank {} checked [{},{},{}], found: {}".format(local_rank,x,y,z,hf['native_fields']['baryon_density'][x1+2, y1+2, z1+2]))
                        hf['native_fields']['baryon_density'][x1+2,y1+2,z1+2] = 1
                    
                        x_plus = None if (x2_edge == 0) else -x2_edge
                        y_plus = None if (y2_edge == 0) else -y2_edge
                        z_plus = None if (z2_edge == 0) else -z2_edge
                        
                        sliced_in = f['Nbody'][:, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge].astype(dtype)
                        sliced_in = np.expand_dims(sliced_in, axis=0) # add batch dim
                        sliced_in = torch.from_numpy(sliced_in)

                        # sliced_in = full[:, :, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge]
                        print("Rank {} received chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in.shape))

                        with torch.no_grad():
                            chunk = model(sliced_in)
                            
                        # un-normalize the NN output and write to file   
                        hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = np.exp(14.*chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus])
                        hf['native_fields']['velocity_x'][x1:x2,y1:y2,z1:z2] = chunk[0, 1, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                        hf['native_fields']['velocity_y'][x1:x2,y1:y2,z1:z2] = chunk[0, 2, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                        hf['native_fields']['velocity_z'][x1:x2,y1:y2,z1:z2] = chunk[0, 3, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                        hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = np.exp(8.*(chunk[0, 4, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5))
                        
                        #print("Chunk [{},{},{}] output shape: {}".format(x,y,z,chunk.shape))

                    if args.skip: break
                if args.skip: break
            if args.skip: break
    

'''
final=torch.empty((1,5,full_dim,full_dim,full_dim), dtype=torch.float32)

if not args.dummy: 
    for x in range(slices):
        x1 = int(x*size)
        x1_edge = trim if (x>0) else 0
        x2 = int(min((x+1)*size,full_dim))
        x2_edge = trim if (x<slices-1) else 0
        for y in range(slices):
            y1 = int(y*size)
            y1_edge = trim if (y>0) else 0
            y2 = int(min((y+1)*size,full_dim))
            y2_edge = trim if (y<slices-1) else 0
            for z in range(slices):
                if not chunk_done[x][y][z]:
                    chunk_done[x][y][z] = True
                    z1 = int(z*size)
                    z1_edge = trim if (z>0) else 0
                    z2 = int(min((z+1)*size,full_dim))
                    z2_edge = trim if (z<slices-1) else 0

                    x_plus = None if (x2_edge == 0) else -x2_edge
                    y_plus = None if (y2_edge == 0) else -y2_edge
                    z_plus = None if (z2_edge == 0) else -z2_edge

                    sliced_in = full[:, :, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge]

                    print("Chunk [{},{},{}] input shape: {}".format(x,y,z,sliced_in.shape))

                    with torch.no_grad():
                        chunk = model(sliced_in)

                    final[:,:,x1:x2,y1:y2,z1:z2] = chunk[:,:,x1_edge:x_plus,y1_edge:y_plus,z1_edge:z_plus]
                    print("Chunk [{},{},{}] output shape: {}".format(x,y,z,chunk.shape))

                if args.skip: break
            if args.skip: break
        if args.skip: break
'''



'''
if  world_rank==0:
    print("output shape: {}".format(final.shape))
'''

if not args.dummy:
    hf.close()
    
maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')