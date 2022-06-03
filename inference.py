import numpy as np
import h5py
import torch
import torch.nn.functional as nn
from torch.nn.parallel import DistributedDataParallel
from utils.YParams import YParams
from networks import UNet as UNet
import argparse
import resource
import os
import h5py
from collections import OrderedDict
import datetime

datapath = '/path/to/normalized/h5'


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/pscratch/sd/c/cjacobus/ml-pm-training-2022/logs/vanilla/4GPU/00/', type=str)
parser.add_argument("--weights", default='training_checkpoints/ckpt.tar', type=str)
parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
parser.add_argument("--config", default='vanilla', type=str)
parser.add_argument("--datapath", default='/pscratch/sd/p/pharring/Nyx_nbod2hydro/normalized_data/test.h5', type=str)
parser.add_argument("--trim", default=64, type=int)
parser.add_argument("--size", default=512, type=int)
parser.add_argument("--flavor", default='vanilla', type=str)
args = parser.parse_args()
params = YParams(os.path.abspath(args.yaml_config), args.config)

size=args.size
trim=args.trim
dtype=np.single
weights_path=os.path.join(args.folder, args.weights)

params.distributed = False
world_rank = 0
local_rank = 0


if 'WORLD_SIZE' in os.environ:
    params.distributed = int(os.environ['WORLD_SIZE']) > 1
    world_size = int(os.environ['WORLD_SIZE'])
else:
    world_size = 1

if params.distributed:
    torch.distributed.init_process_group(backend='gloo', init_method='env://')
    world_rank = torch.distributed.get_rank()
    local_rank = int(os.environ['LOCAL_RANK'])

if  world_rank==0:
    print("Initializing model...")
model = UNet.UNet(params)
if  world_rank==0:
    print("Initialized model [✓]")

'''
if params.distributed:
    model = DistributedDataParallel(model)
'''

if  world_rank==0:
    print("Loading checkpoint: {}".format(weights_path))
    
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
model.eval()

if  world_rank==0:
    print("Loaded model checkpoint [✓]")
    print("Loading data: {}".format(args.datapath))
    
cosmo = [0,0,0,0,0]

with h5py.File(args.datapath, 'r') as f:
    full = f['Nbody'][:,:,:,:].astype(dtype)
    
    dom_size = f['domain'].attrs['size']
    dom_shape = f['domain'].attrs['shape']
    
    for k in f['universe'].attrs.keys():
        cosmo[k] = f['universe'].attrs[k]
    
    full = np.expand_dims(full, axis=0) # add batch dim
    full = torch.from_numpy(full)
    full_dim=full.shape[3]
    slices=int(full_dim/size)
    
    if  world_rank==0:
        print("Loaded data [✓]")
        print("Full shape: {}".format(full.shape))
        print("Dimension: {}".format(full_dim))
        print("Trimmed chunk size: {}".format(size))
        print("Trim: {}".format(trim))
        print("Slices: {}".format(slices))
        print("Sending {} chunks individually...".format(slices**3))
        
        
    #sliced_in = [[['#' for x in range(slices)] for y in range(slices)] for z in range(slices)]
    final=torch.empty((1,5,full_dim,full_dim,full_dim), dtype=torch.float32)
    
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
                
                x_plus = None if (x2_edge == 0) else -x2_edge
                y_plus = None if (y2_edge == 0) else -y2_edge
                z_plus = None if (z2_edge == 0) else -z2_edge
                
                sliced_in = full[:, :, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge]
                print("Chunk [{},{},{}] input shape: {}".format(x,y,z,sliced_in.shape))
                
                with torch.no_grad():
                    chunk = model(sliced_in)
                    #chunk = torch.from_numpy(chunk)
                
                final[:,:,x1:x2,y1:y2,z1:z2] = chunk[:,:,x1_edge:x_plus,y1_edge:y_plus,z1_edge:z_plus]
                print("Chunk [{},{},{}] output shape: {}".format(x,y,z,chunk.shape))
                #break 



    #sliced_out = [[['#' for x in range(slices)] for y in range(slices)] for z in range(slices)]

    #final=np.ndarray(shape=(1,5,full_dim,full_dim,full_dim), dtype=float)


if  world_rank==0:
    print("output shape: {}".format(final.shape))

hf = h5py.File("{}inference_{}_size_{}_trim_{}.h5".format(args.folder,args.flavor,size,trim), 'w')
hf.attrs['format'] = "nyx-lyaf"
hf.attrs['chunk'] = size
hf.attrs['trim'] = trim
hf.attrs['flavor'] = args.flavor

dom = hf.create_group("domain")
dom.attrs['size'] = dom_size
dom.attrs['shape'] = dom_shape

uni = hf.create_group("universe")
uni.attrs['hubble'] = cosmo[0]
uni.attrs['omega_b'] = cosmo[1]
uni.attrs['omega_l'] = cosmo[2]
uni.attrs['omega_m'] = cosmo[3]
uni.attrs['redshift'] = cosmo[4]

# un-normalize the NN output and write to file
hf.create_dataset("native_fields/baryon_density", data=np.exp(14.*final[0,0,:,:,:]))
hf.create_dataset("native_fields/velocity_x", data=final[0,1,:,:,:]*9e7)
hf.create_dataset("native_fields/velocity_y", data=final[0,2,:,:,:]*9e7)
hf.create_dataset("native_fields/velocity_z", data=final[0,3,:,:,:]*9e7)
hf.create_dataset("native_fields/temperature", data=np.exp(8.*(final[0,4,:,:,:] + 1.5)))
hf.close()
    
maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')