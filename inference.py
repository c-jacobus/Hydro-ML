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
from collections import OrderedDict

datapath = '/path/to/normalized/h5'


parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/pscratch/sd/c/cjacobus/ml-pm-training-2022/logs/vanilla/4GPU/00/', type=str)
parser.add_argument("--weights", default='training_checkpoints/ckpt.tar', type=str)
parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str)
parser.add_argument("--config", default='vanilla', type=str)
parser.add_argument("--datapath", default='/pscratch/sd/p/pharring/Nyx_nbod2hydro/normalized_data/test.h5', type=str)
args = parser.parse_args()
params = YParams(os.path.abspath(args.yaml_config), args.config)

size=1024
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
    print("Loading data: {}".format(args.datapath))
with h5py.File(args.datapath, 'r') as f:
    sample = f['Nbody'][:,:size,:size,:size].astype(dtype)
    sample = np.expand_dims(sample, axis=0) # add batch dim
    sample = torch.from_numpy(sample)
if  world_rank==0:
    print("Loaded data [✓]")
    print("Sample shape: {}".format(sample.shape))

    
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
    print("Loaded ckeckpoint [✓]")
    print("Sending sample to model...")

with torch.no_grad():
    x = model(sample)
    x = x.numpy()

if  world_rank==0:
    print("output shape: {}".format(x.shape))
    np.save(args.folder+"eval_out.npy", x)
    maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]
if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')