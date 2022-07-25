import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal

filters = {
            "relu": nn.ReLU(inplace=True),
            "sigmoid": nn.Sigmoid(),
            "leaky": nn.LeakyReLU(inplace=True),
            "self": nn.Identity(),
            }

a = np.array([1., 4., 6., 4., 1.])
blur = torch.Tensor(a[:,None,None]*a[None,:,None]*a[None,None,:])
blur = blur/torch.sum(blur)

blur = blur.view(1, 1, 5, 5, 5)

class BlurPool(nn.Module):
    def __init__(self, channels, filt_size=5, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:,None,None]*a[None,:,None]*a[None,None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[:,:,:].repeat((self.channels,1,1,1,1)))

        #self.pad = nn.ReflectionPad2d(self.pad_sizes)

    def forward(self, inp):
        return F.conv3d(inp, self.filt, stride=self.stride, padding=2, groups=inp.shape[1])
        
def down_conv(in_channels, out_channels, filter, blurry):
    if blurry:        
        return nn.Sequential(
            BlurPool(in_channels, filt_size=5, stride=2),
            nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2),
            filters[filter],
        )
    
    else:
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1),
            filters[filter],
        )   

'''
def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0),
        nn.ReLU(inplace=True),
    )

'''

def up_conv(in_channels, out_channels, filter, Upsample=False):
    if Upsample:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, out_channels, 3, stride=1, padding=1),
            filters[filter],
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0),
            filters[filter],
        )


def inverse_transf(x):
    return torch.exp(14.*x)


def loss_func(gen_output, target, lambda_chi, lambda_spec, trim):
    
    
    gen = gen_output[:,0, trim:-trim, trim:-trim, trim:-trim]
    tar = target[:,0, trim:-trim, trim:-trim, trim:-trim]
   
    # first part of the loss
    l1_loss = nn.functional.l1_loss(gen, tar)
    # print(f'l1_loss = {l1_loss}')
    
    if lambda_chi > 0. or lambda_spec > 0.:
        gen = gen_output.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
        tar = target.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
        gen = np.min(gen,1)
        tar = np.squeeze(tar, axis=1)

    chi = 0
    if lambda_chi > 0.:
        tar_hist, bin_edges = np.histogram(tar[:,:,:,:], bins=50)
        gen_hist, _ = np.histogram(gen[:,:,:,:], bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        tar_clean = tar_hist[tar_hist >= 1.]
        gen_clean = gen_hist[tar_hist >= 1.]        
        
        chi = np.sum(np.divide(np.power(tar_clean - gen_clean, 2.0), tar_clean))
        #print(f'chi = {chi}')
        
    spec = 0
    if lambda_spec > 0.:
        tar_mean = np.mean(tar)
        tar_over = (tar-tar_mean)/tar_mean
        
        
        gen_mean = np.mean(gen)
        gen_over = (gen-gen_mean)/gen_mean
        
        #print(f'tar_over: {tar_over.shape}')
        #print(f'gen_over: {gen_over.shape}')
        
        mean_axis = [(1,2,3), (0,2,3), (0,1,3), (0,1,2)]
        
        for i in range(1,4):
            
            f, Pxx_tar = signal.periodogram(tar_over, fs=1, scaling='spectrum', axis=i) # spectrum || density
            Pxx_tar = np.mean(Pxx_tar, axis=mean_axis[i])
            #print(f'Pxx_tar: {Pxx_tar.shape}')

            f, Pxx_gen = signal.periodogram(gen_over, fs=1, scaling='spectrum', axis=i) # spectrum || density
            Pxx_gen = np.mean(Pxx_gen, axis=mean_axis[i])
            #print(f'Pxx_gen: {Pxx_gen.shape}')

            spec += np.sum(np.divide(np.power(Pxx_gen[:] - Pxx_tar[:], 2.0), Pxx_tar[:]))
            
        spec = spec/3
        #print(f'L_spec = {lambda_spec * spec}')
        #print(f'L_chi = {lambda_chi * chi}')
    
    return l1_loss + (lambda_chi * chi) + (lambda_spec * spec) , l1_loss , (lambda_chi * chi) , (lambda_spec * spec)


@torch.jit.script
def loss_func_opt(gen_output: torch.Tensor, target: torch.Tensor, lambda_chi: float):

    # first part of the loss
    l1_loss = torch.mean(torch.abs(gen_output - target))

    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output[:,0,:,:,:])
    orig_tar = inverse_transf(target[:,0,:,:,:])
    orig_l1_loss = torch.mean(torch.abs(orig_gen - orig_tar))
    return l1_loss + lambda_chi * orig_l1_loss  


@torch.jit.script
def loss_func_opt_final(gen_output: torch.Tensor, target: torch.Tensor, lambda_chi: torch.Tensor):

    # first part of the loss
    l1_loss = torch.abs(gen_output - target)
    
    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output)
    orig_tar = inverse_transf(target)
    orig_l1_loss = torch.abs(orig_gen - orig_tar)

    # combine
    loss = l1_loss + lambda_chi * orig_l1_loss
    
    return torch.mean(loss)


class UNet(nn.Module):

    def __init__(self, params):
        super().__init__()
        
        
        self.full_scale = params.full_scale
        self.conv_down1 = down_conv(params.N_in_channels, 64, params.down_filter, params.blur)
        self.conv_down2 = down_conv(64, 128, params.down_filter, params.blur)
        self.conv_down3 = down_conv(128, 256, params.down_filter, params.blur)
        self.conv_down4 = down_conv(256, 512, params.down_filter, params.blur)        
        self.conv_down5 = down_conv(512, 512, params.down_filter, params.blur)
        if self.full_scale:
          self.conv_down6 = down_conv(512, 512, params.down_filter, params.blur)

          self.conv_up6 = up_conv(512, 512, params.up_filter, params.upsample)
          #self.conv_up5 = up_conv(512+512, 512)
          self.conv_up5 = up_conv(512, 512, params.up_filter, params.upsample)
        else:
          self.conv_up5 = up_conv(512, 512, params.up_filter, params.upsample)
        #self.conv_up4 = up_conv(512+512, 256)
        self.conv_up4 = up_conv(512, 256, params.up_filter, params.upsample)
        self.conv_up3 = up_conv(256+256, 128, params.up_filter, params.upsample)
        self.conv_up2 = up_conv(128+128, 64, params.up_filter, params.upsample)
        #self.conv_last = nn.ConvTranspose3d(64+64, params.N_out_channels, 4, stride=2, padding=1, output_padding=0)
        
        if params.upsample:
            self.conv_last = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), nn.Conv3d(64+64, params.N_out_channels, 3, stride=1, padding=1))
            
        else:
            self.conv_last = nn.ConvTranspose3d(64+64, params.N_out_channels, 4, stride=2, padding=1, output_padding=0)
            
        self.last = filters[params.last_filter]
        
      
        
    def forward(self, x):
        describe = False
        conv1 = self.conv_down1(x) # 64
        
        if describe: print(f'down1 = {conv1.size()}')
        conv2 = self.conv_down2(conv1) # 128
        if describe: print(f'down2 = {conv2.size()}')
        conv3 = self.conv_down3(conv2) # 256
        if describe: print(f'down3 = {conv3.size()}')
        conv4 = self.conv_down4(conv3) # 512
        if describe: print(f'down4 = {conv4.size()}')
        conv5 = self.conv_down5(conv4) # 512
        if describe: print(f'down5 = {conv5.size()}')
        if self.full_scale:
            conv6 = self.conv_down6(conv5) # 512
            if describe: print(f'down6 = {conv6.size()}')
        
            x = self.conv_up6(conv6) # 512
            if describe: print(f'up6 = {x.size()}')
            #x = torch.cat([x, conv5], dim=1)
        else:
            x = conv5
        x = self.conv_up5(x) # 512
        if describe: print(f'up5 = {x.size()}')
        #x = torch.cat([x, conv4], dim=1)
        x = self.conv_up4(x) # 256
        if describe: print(f'up4 = {x.size()}')
        if describe: print(f'down3 = {conv3.size()}')
        x = torch.cat([x, conv3], dim=1)
        x = self.conv_up3(x) # 128
        x = torch.cat([x, conv2], dim=1)
        x = self.conv_up2(x) # 64
        x = torch.cat([x, conv1], dim=1)
        x = self.conv_last(x) # 5
        out =  x = self.last(x)
        return out

    def get_weights_function(self, params):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])
                if params['conv_bias'] is not None:
                    m.bias.data.fill_(params['conv_bias'])
        return weights_init
 
