import sys
import os
import time
import numpy as np
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from apex import amp, optimizers
#from apex.parallel import DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing

import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader import get_data_loader_distributed
from utils.plotting import generate_images, meanL1

from torch.cuda.amp import autocast, GradScaler

from utils import get_data_loader_distributed, lr_schedule
from networks import UNet

import apex.optimizers as aoptim

def train(params, args, local_rank, world_rank, world_size):
    # set device and benchmark mode
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda:%d'%local_rank)

    # get data loader
    logging.info('rank %d, begin data loader init'%world_rank)
    train_data_loader, val_data_loader = get_data_loader_distributed(params, world_rank, device.index)
    logging.info('rank %d, data loader initialized with config %s'%(world_rank, params.data_loader_config))

    # create model
    model = UNet.UNet(params).to(device)
    model.apply(model.get_weights_function(params.weight_init))
  
    if params.enable_amp:
        scaler = GradScaler()
    if params.distributed and not args.noddp:
        if args.disable_broadcast_buffers: 
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                    bucket_cap_mb=args.bucket_cap_mb,
                                    broadcast_buffers=False,
                                    gradient_as_bucket_view=True)
        else:
            model = DistributedDataParallel(model, device_ids=[local_rank],
                                    bucket_cap_mb=args.bucket_cap_mb)

    if params.enable_apex:
        optimizer = aoptim.FusedAdam(model.parameters(), lr = params.lr_schedule['start_lr'],
                                 adam_w_mode=False, set_grad_none=True)
    else:
        optimizer = optim.Adam(model.parameters(), lr = params.lr_schedule['start_lr'])

    if params.enable_jit:
        torch._C._jit_set_nvfuser_enabled(True)
        torch._C._jit_set_texpr_fuser_enabled(False)
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        #torch._C._jit_set_profiling_executor(True)
        #torch._C._jit_set_profiling_mode(True)
        #torch._C._jit_set_bailout_depth(20)
        model_handle = model.module if (params.distributed and not args.noddp) else model
        model_handle = torch.jit.script(model_handle)  

    # select loss function
    if params.enable_jit:
        loss_func = UNet.loss_func_opt_final
        lambda_rho = torch.zeros((1,5,1,1,1), dtype=torch.float32).to(device)
        lambda_rho[:,0,:,:,:] = params.lambda_rho
    else:
        loss_func = UNet.loss_func
        lambda_rho = params.lambda_rho

    # start training
    iters = 0
    startEpoch = 0
    params.lr_schedule['tot_steps'] = params.num_epochs*(params.Nsamples//params.global_batch_size)

    if world_rank==0: 
        logging.info("Starting Training Loop...")

  # Log initial loss on train and validation to tensorboard
    if not args.enable_benchy:
        with torch.no_grad():
            inp, tar = map(lambda x: x.to(device), next(iter(train_data_loader)))
            tr_loss = loss_func(model(inp), tar, lambda_rho)
            inp, tar = map(lambda x: x.to(device), next(iter(val_data_loader)))
            val_loss= loss_func(model(inp), tar, lambda_rho)
            if params.distributed:
                torch.distributed.all_reduce(tr_loss)
                torch.distributed.all_reduce(val_loss)
            if world_rank==0:
                tboard_writer.add_scalar('Loss/train', tr_loss.item()/world_size, 0)
                tboard_writer.add_scalar('Loss/valid', val_loss.item()/world_size, 0)

    iters = 0
    t1 = time.time()
    for epoch in range(startEpoch, startEpoch+params.num_epochs):
        start = time.time()
        tr_loss = []
        tr_time = 0.
        dat_time = 0.
        log_time = 0.

        model.train()
        step_count = 0
        for i, data in enumerate(train_data_loader, 0):
            iters += 1
            dat_start = time.time()
            inp, tar = map(lambda x: x.to(device), data)
            tr_start = time.time()
            b_size = inp.size(0)
      
        lr_schedule(optimizer, iters, global_bs=params.global_batch_size, base_bs=params.base_batch_size, **params.lr_schedule)
        optimizer.zero_grad()
        with autocast(params.enable_amp):
            gen = model(inp)
            loss = loss_func(gen, tar, lambda_rho)
            tr_loss.append(loss.item())

        if params.enable_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        tr_end = time.time()
        tr_time += tr_end - tr_start
        dat_time += tr_start - dat_start
        step_count += 1

        end = time.time()
        if world_rank==0:
            logging.info('Time taken for epoch {} is {} sec, avg {} samples/sec'.format(epoch + 1, end-start,
                                                                                      (step_count * params["global_batch_size"])/(end-start)))
            logging.info('  Avg train loss=%f'%np.mean(tr_loss))
            tboard_writer.add_scalar('Loss/train', np.mean(tr_loss), iters)
            tboard_writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iters)
            tboard_writer.add_scalar('Avg iters per sec', step_count/(end-start), iters)

            log_start = time.time()
            gens = []
            tars = []
            with torch.no_grad():
                for i, data in enumerate(train_data_loader, 0):
                    if i>=16:
                        break
                    inp, tar = map(lambda x: x.to(device), data)
                    gen = model(inp)
                    gens.append(gen.detach().cpu().numpy())
                    tars.append(tar.detach().cpu().numpy())
            gens = np.concatenate(gens, axis=0)
            tars = np.concatenate(tars, axis=0)
    
            # Scalars
            tboard_writer.add_scalar('G_loss', loss.item(), iters)

            # Plots
            fig, chi, L1score = meanL1(gens, tars)
            tboard_writer.add_figure('pixhist', fig, iters, close=True)
            tboard_writer.add_scalar('Metrics/chi', chi, iters)
            tboard_writer.add_scalar('Metrics/rhoL1', L1score[0], iters)
            tboard_writer.add_scalar('Metrics/vxL1', L1score[1], iters)
            tboard_writer.add_scalar('Metrics/vyL1', L1score[2], iters)
            tboard_writer.add_scalar('Metrics/vzL1', L1score[3], iters)
            tboard_writer.add_scalar('Metrics/TL1', L1score[4], iters)
            
            fig = generate_images(inp.detach().cpu().numpy()[0], gens[-1], tars[-1])
            tboard_writer.add_figure('genimg', fig, iters, close=True)
    
    val_start = time.time()
    val_loss = []
    model.eval()
    if not args.enable_benchy:
        with torch.no_grad():
            for i, data in enumerate(val_data_loader, 0):
                with autocast(params.enable_amp):
                    inp, tar = map(lambda x: x.to(device), data)
                    gen = model(inp)
                    loss = loss_func(gen, tar, lambda_rho)
                    if params.distributed:
                        torch.distributed.all_reduce(loss)
                    val_loss.append(loss.item()/world_size)
        val_end = time.time()
        if world_rank==0:
            logging.info('  Avg val loss=%f'%np.mean(val_loss))
            logging.info('  Total validation time: {} sec'.format(val_end - val_start)) 
            tboard_writer.add_scalar('Loss/valid', np.mean(val_loss), iters)
            tboard_writer.flush()

    t2 = time.time()
    tottime = t2 - t1



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str, help='tag for indexing the current experiment')
    parser.add_argument("--yaml_config", default='./config/UNet.yaml', type=str, help='path to yaml file containing training configs')
    parser.add_argument("--config", default='base', type=str, help='name of desired config in yaml file')
    parser.add_argument("--enable_amp", action='store_true', help='enable automatic mixed precision')
    parser.add_argument("--enable_apex", action='store_true', help='enable apex fused Adam optimizer')
    parser.add_argument("--enable_jit", action='store_true', help='enable JIT compilation')
    parser.add_argument("--enable_benchy", action='store_true', help='enable benchy tool usage')
    parser.add_argument("--data_loader_config", default=None, type=str,
                      choices=['synthetic', 'inmem', 'lowmem', 'dali-lowmem'],
                      help="dataloader configuration. choices: 'synthetic', 'inmem', 'lowmem', 'dali-lowmem'")
    parser.add_argument("--local_batch_size", default=None, type=int, help='local batchsize (manually override global_batch_size config setting)')
    parser.add_argument("--num_epochs", default=None, type=int, help='number of epochs to run')
    parser.add_argument("--num_data_workers", default=None, type=int, help='number of data workers for data loader')
    parser.add_argument("--bucket_cap_mb", default=25, type=int, help='max message bucket size in mb')
    parser.add_argument("--disable_broadcast_buffers", action='store_true', help='disable syncing broadcasting buffers')
    parser.add_argument("--noddp", action='store_true', help='disable DDP communication')
    args = parser.parse_args()

    run_num = args.run_num

    params = YParams(os.path.abspath(args.yaml_config), args.config)

    # Update config with modified args
    params.update({"enable_amp" : args.enable_amp,
                    "enable_apex" : args.enable_apex,
                    "enable_jit" : args.enable_jit,
                    "enable_benchy" : args.enable_benchy})

    if args.data_loader_config:
        params.update({"data_loader_config" : args.data_loader_config})

    if args.num_epochs:
        params.update({"num_epochs" : args.num_epochs})

    if args.num_data_workers:
        params.update({"num_data_workers" : args.num_data_workers})

    params.distributed = False
    if 'WORLD_SIZE' in os.environ:
        params.distributed = int(os.environ['WORLD_SIZE']) > 1
        world_size = int(os.environ['WORLD_SIZE'])
    else:
        world_size = 1

    world_rank = 0
    local_rank = 0
    if params.distributed:
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])

    if args.local_batch_size:
        # Manually override batch size
        params.local_batch_size = args.local_batch_size
        params.update({"global_batch_size" : world_size*args.local_batch_size})
    else:
        # Compute local batch size based on number of ranks
        params.local_batch_size = params.global_batch_size//world_size

    # Set up directory
    baseDir = params.expdir
    expDir = os.path.join(baseDir, args.config+'/%dGPU/'%(world_size)+str(run_num)+'/')
    if  world_rank==0:
        if not os.path.isdir(expDir):
            os.makedirs(expDir)
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        params.log()
        tboard_writer = SummaryWriter(log_dir=os.path.join(expDir, 'logs/'))

    params.experiment_dir = os.path.abspath(expDir)

    train(params, args, local_rank, world_rank, world_size)
    if params.distributed:
        torch.distributed.barrier()
    if world_rank == 0:
        tboard_writer.flush()
        tboard_writer.close()
    logging.info('DONE ---- rank %d'%world_rank)