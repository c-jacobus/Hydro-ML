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
parser.add_argument("--target_path", default='/pscratch/sd/p/pharring/Nyx_nbod2hydro/raw_data/test_hydro.h5', type=str)
parser.add_argument("--pred_path", default='/pscratch/sd/c/cjacobus/ml-pm-training-2022/logs/vanilla/4GPU/00/inference_trim_64_1654216439.h5', type=str)
parser.add_argument("--mode", default='error', type=str)
args = parser.parse_args()

dtype=np.single

target = h5py.File(args.target_path, 'r')

tar_bar = target['native_fields']['baryon_density'][:,:,:].astype(dtype)
tar_vx = target['native_fields']['velocity_x'][:,:,:].astype(dtype)
tar_vy = target['native_fields']['velocity_y'][:,:,:].astype(dtype)
tar_vz = target['native_fields']['velocity_z'][:,:,:].astype(dtype)
tar_temp = target['native_fields']['temperature'][:,:,:].astype(dtype)

print("Target type: {}".format(type(tar_bar)))     
print("Target shape: {}".format(tar_bar.shape))

predict = h5py.File(args.pred_path, 'r')
form = predict.attrs['format']
size = predict.attrs['chunk']
trim = predict.attrs['trim']
flavor = predict.attrs['flavor']

pred_bar = predict['native_fields']['baryon_density'][:,:,:].astype(dtype)
pred_vx = predict['native_fields']['velocity_x'][:,:,:].astype(dtype)
pred_vy = predict['native_fields']['velocity_y'][:,:,:].astype(dtype)
pred_vz = predict['native_fields']['velocity_z'][:,:,:].astype(dtype)
pred_temp = predict['native_fields']['temperature'][:,:,:].astype(dtype)
print("Prediction type: {}".format(type(pred_bar)))     
print("Prediction shape: {}".format(pred_bar.shape))


print(tar_bar[0,0,0])
print(pred_bar[0,0,0])
print(tar_bar[0,0,0]-pred_bar[0,0,0])

if args.mode=='error':
    err_bar = np.divide(np.subtract(tar_bar, pred_bar), tar_bar+1e-2)
    err_vx = np.divide(np.subtract(tar_vx, pred_vx), tar_vx+1e-2)
    err_vy = np.divide(np.subtract(tar_vy, pred_vy), tar_vy+1e-2)
    err_vz = np.divide(np.subtract(tar_vz, pred_vz), tar_vz+1e-2)
    err_temp = np.divide(np.subtract(tar_temp, pred_temp), tar_temp+1e-2)
else:
    err_bar = np.subtract(tar_bar, pred_bar)
    err_vx = np.subtract(tar_vx, pred_vx)
    err_vy = np.subtract(tar_vy, pred_vy)
    err_vz = np.subtract(tar_vz, pred_vz)
    err_temp = np.subtract(tar_temp, pred_temp)


hf = h5py.File("{}{}_{}_size_{}_trim_{}.h5".format(args.folder,args.mode,flavor,size,trim), 'w')
hf.attrs['format'] = form
hf.attrs['mode'] = args.mode
hf.attrs['chunk'] = size
hf.attrs['trim'] = trim
hf.attrs['flavor'] = flavor

dom = hf.create_group("domain")
dom.attrs['size'] = [21.09375,21.09375,21.09375]
dom.attrs['shape'] = [1024,1024,1024]

hf.create_dataset("native_fields/baryon_density", data=err_bar)
hf.create_dataset("native_fields/velocity_x", data=err_vx)
hf.create_dataset("native_fields/velocity_y", data=err_vy)
hf.create_dataset("native_fields/velocity_z", data=err_vy)
hf.create_dataset("native_fields/temperature", data=err_temp)
hf.close()


print('DONE')
