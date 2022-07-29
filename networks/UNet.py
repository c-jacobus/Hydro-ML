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
    
def dense_conv(in_channels, out_channels, filter):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 5, stride=1, padding=2),
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



class LossFunc(nn.Module):
    def __init__(self, lambda_chi, lambda_spec, trim):
        super().__init__()
        
        device = torch.cuda.current_device()
        mask = np.zeros((128, 128, 128))
        rad = 56
        cent = 64
        for x in range(128):
            for y in range(128):
                for z in range(128):
                    if ((x-cent)**2 + (y-cent)**2 + (z-cent)**2) <= rad**2:
                        mask[x,y,z] = 1
        
        if trim > 0:
            self.mask = torch.from_numpy(mask[None, trim:-trim, trim:-trim, trim:-trim]).to(device)
        else:
            self.mask = torch.from_numpy(mask[None, :, :, :]).to(device)
            
        self.lambda_chi = lambda_chi
        self.lambda_spec = lambda_spec
        self.trim = trim
        
    def forward(self, gen_output, target, metric=False):
        lambda_chi = self.lambda_chi 
        lambda_spec = self.lambda_spec
        trim = self.trim
        
        if trim > 0:
            gen = gen_output[:,0, trim:-trim, trim:-trim, trim:-trim]
            tar = target[:,0, trim:-trim, trim:-trim, trim:-trim]
        else:
            gen = gen_output
            tar = target
            
        # basic L1 loss
        l1_loss = nn.functional.l1_loss(gen, tar)
        
        #fft loss
        fft_loss = 0
        if lambda_spec > 0. or metric:
            
            tar_mean = torch.mean(tar)
            tar_over = (tar-tar_mean)/tar_mean

            gen_mean = torch.mean(gen)
            gen_over = (gen-gen_mean)/gen_mean

            # Take FFT and return L1 loss
            #tar_fft = torch.fft.fftn(tar_over, norm='ortho', dim=(1,2,3))
            tar_fft = torch.fft.fftn(tar_over, norm='ortho', dim=(1,2,3))
            tar_fft = torch.fft.fftshift(tar_fft)
            tar_fft_r = torch.mul(tar_fft.real, self.mask)
            tar_fft_i = torch.mul(tar_fft.imag, self.mask)

            gen_fft = torch.fft.fftn(gen_over, norm='ortho', dim=(1,2,3))
            gen_fft = torch.fft.fftshift(gen_fft)
            gen_fft_r = torch.mul(gen_fft.real, self.mask)
            gen_fft_i = torch.mul(gen_fft.imag, self.mask)
            
            

            fft_loss += nn.functional.l1_loss(torch.log1p(gen_fft_r.abs()), torch.log1p(tar_fft_r.abs()))
            fft_loss += nn.functional.l1_loss(torch.log1p(gen_fft_i.abs()), torch.log1p(tar_fft_i.abs()))

        chi = 0
        if lambda_chi > 0. or metric:
            nbins = 20
            tar_hist = torch.histc(tar, bins=nbins, min=0, max=1) #torch.histogram(tar, bins=20, range=(0,1), weight=None, density=True)
            gen_hist = torch.histc(gen, bins=nbins, min=0, max=1) #torch.histogram(gen, bins=Bins, weight=None, density=True)
            
            chi = torch.sum(torch.div(torch.pow(tar_hist - gen_hist, 2), tar_hist))
            '''
            gen = gen_output.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
            tar = target.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
            gen = np.min(gen,1)
            tar = np.squeeze(tar, axis=1)
            tar_hist, bin_edges = np.histogram(tar[:,:,:,:], bins=50)
            gen_hist, _ = np.histogram(gen[:,:,:,:], bins=bin_edges)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            tar_clean = tar_hist[tar_hist >= 1.]
            gen_clean = gen_hist[tar_hist >= 1.]        

            chi = np.sum(np.divide(np.power(tar_clean - gen_clean, 2.0), tar_clean))
            #print(f'chi = {chi}')
            '''


        return l1_loss + (lambda_chi * chi) + (lambda_spec * fft_loss) , l1_loss , chi , fft_loss
        
'''
def loss_func(gen_output, target, lambda_chi, lambda_spec, trim, device, metric=False):
    
    
    if trim > 0:
        gen = gen_output[:,0, trim:-trim, trim:-trim, trim:-trim]
        tar = target[:,0, trim:-trim, trim:-trim, trim:-trim]
    else:
        gen = gen_output
        tar = target
        
    # first part of the loss
    l1_loss = nn.functional.l1_loss(gen, tar)
    # print(f'l1_loss = {l1_loss}')
    
    fft_loss = 0
    if lambda_spec > 0. or metric:
        mask_128 = torch.from_numpy(mask[None, :, :, :]).to(device)
        
        tar_mean = torch.mean(tar)
        tar_over = (tar-tar_mean)/tar_mean
        
        gen_mean = torch.mean(gen)
        gen_over = (gen-gen_mean)/gen_mean
        
        # Take FFT and return L1 loss
        #tar_fft = torch.fft.fftn(tar_over, norm='ortho', dim=(1,2,3))
        tar_fft = torch.fft.fftn(target, norm='ortho', dim=(1,2,3))
        tar_fft = torch.fft.fftshift(tar_fft)
        tar_fft = torch.mul(tar_fft, mask_128)
        
        gen_fft = torch.fft.fftn(gen_output, norm='ortho', dim=(1,2,3))
        gen_fft = torch.fft.fftshift(gen_fft)
        gen_fft = torch.mul(gen_fft, mask_128)
        
        fft_loss = nn.functional.l1_loss(torch.log1p(gen_fft.abs()), torch.log1p(tar_fft.abs()))
        
        
    

    chi = 0
    if lambda_chi > 0. or metric:
        gen = gen_output.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
        tar = target.detach().cpu().numpy()[:,:, trim:-trim, trim:-trim, trim:-trim]
        gen = np.min(gen,1)
        tar = np.squeeze(tar, axis=1)
        tar_hist, bin_edges = np.histogram(tar[:,:,:,:], bins=50)
        gen_hist, _ = np.histogram(gen[:,:,:,:], bins=bin_edges)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        tar_clean = tar_hist[tar_hist >= 1.]
        gen_clean = gen_hist[tar_hist >= 1.]        
        
        chi = np.sum(np.divide(np.power(tar_clean - gen_clean, 2.0), tar_clean))
        #print(f'chi = {chi}')
        
    
    if metric:
        return l1_loss + (lambda_chi * chi) + (lambda_spec * fft_loss) , l1_loss , chi , fft_loss
    else:
        #return torch.add(l1_loss, fft_loss, alpha=lambda_spec)
        return l1_loss + lambda_spec*fft_loss
'''


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
        self.dense = params.dense
        
        if not self.dense:
            self.conv_down1 = down_conv(params.N_in_channels, 64, params.down_filter, params.blur)
            self.conv_down2 = down_conv(64, 128, params.down_filter, params.blur)
            self.conv_down3 = down_conv(128, 256, params.down_filter, params.blur)
            self.conv_down4 = down_conv(256, 512, params.down_filter, params.blur)        
            self.conv_down5 = down_conv(512, 512, params.down_filter, params.blur)

            if self.full_scale:
                self.conv_down6 = down_conv(512, 512, params.down_filter, params.blur)

                self.conv_up6 = up_conv(512, 512, params.up_filter, params.upsample)
                self.conv_up5 = up_conv(512, 512, params.up_filter, params.upsample)
            else:
                self.conv_up5 = up_conv(512, 512, params.up_filter, params.upsample)
            self.conv_up4 = up_conv(512, 256, params.up_filter, params.upsample)
            self.conv_up3 = up_conv(256+256, 128, params.up_filter, params.upsample)
            self.conv_up2 = up_conv(128+128, 64, params.up_filter, params.upsample)
            
        else:
            self.d_conv_down0 = dense_conv(params.N_in_channels, 32, params.down_filter)
            self.conv_down1 = down_conv(params.N_in_channels+32, 64, params.down_filter, params.blur)
            
            self.d_conv_down1 = dense_conv(64, 32, params.down_filter)
            self.conv_down2 = down_conv(64+32, 128, params.down_filter, params.blur)
            
            self.d_conv_down2 = dense_conv(128, 64, params.down_filter)
            self.conv_down3 = down_conv(128+64, 256, params.down_filter, params.blur)
            
            self.d_conv_down3 = dense_conv(256, 128, params.down_filter)
            self.conv_down4 = down_conv(256+128, 512, params.down_filter, params.blur)
            
            self.d_conv_down4 = dense_conv(512, 256, params.down_filter)
            self.conv_down5 = down_conv(512+256, 1024, params.down_filter, params.blur)
            
            self.d_conv_down5 = dense_conv(1024, 512, params.down_filter)
            self.conv_up5 = up_conv(1024+512, 512, params.up_filter, params.upsample)
            
            self.d_conv_up4 = dense_conv(512, 256, params.down_filter)
            self.conv_up4 = up_conv(512+256, 256, params.up_filter, params.upsample)
            
            self.d_conv_up3 = dense_conv(256+(256+128), 128, params.down_filter)
            self.conv_up3 = up_conv(768, 128, params.up_filter, params.upsample)
            
            self.d_conv_up2 = dense_conv(128+(128+64), 64, params.down_filter)
            self.conv_up2 = up_conv(384, 64, params.up_filter, params.upsample)
            
            self.d_conv_up1 = dense_conv(64+96, 64, params.down_filter)
        
        sub_ult = 224 if self.dense else 128 
        if params.upsample:
            self.conv_last = nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True), nn.Conv3d(sub_ult, params.N_out_channels, 3, stride=1, padding=1))
            
        else:
            self.conv_last = nn.ConvTranspose3d(sub_ult, params.N_out_channels, 4, stride=2, padding=1, output_padding=0)
            
        self.last = filters[params.last_filter]
        
      
        
    def forward(self, x):
        describe = False
        
        if not self.dense:
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
            x = torch.cat([x, conv3], dim=1)
            x = self.conv_up3(x) # 128
            if describe: print(f'up3 = {x.size()}')
            x = torch.cat([x, conv2], dim=1)
            x = self.conv_up2(x) # 64
            if describe: print(f'up2 = {x.size()}')
            x = torch.cat([x, conv1], dim=1)
            x = self.conv_last(x) # 1
            if describe: print(f'up1 = {x.size()}')
            out =  x = self.last(x)
            if describe: print(f'out = {x.size()}')
            return out
        
        else:
            if describe: print(f'd0 x = {x.size()}') #5
            xd = self.d_conv_down0(x)
            if describe: print(f'd0 xd = {xd.size()}') #32
            down0 = torch.cat([x, xd], dim=1)
            
            x = self.conv_down1(down0)
            if describe: print(f'd1 x = {x.size()}') #64
            xd = self.d_conv_down1(x)
            if describe: print(f'd2 xd = {xd.size()}') #32
            down1 = torch.cat([x, xd], dim=1)
            
            x = self.conv_down2(down1)
            if describe: print(f'd2 x = {x.size()}') #128
            xd = self.d_conv_down2(x)
            if describe: print(f'd2 xd = {xd.size()}') #64
            down2 = torch.cat([x, xd], dim=1)
            
            x = self.conv_down3(down2)
            if describe: print(f'd3 x = {x.size()}') #256
            xd = self.d_conv_down3(x)
            if describe: print(f'd3 xd = {xd.size()}') #128
            down3 = torch.cat([x, xd], dim=1)
            
            x = self.conv_down4(down3)
            if describe: print(f'd4 x = {x.size()}') #512
            xd = self.d_conv_down4(x)
            if describe: print(f'd0 xd = {xd.size()}') #256
            down4 = torch.cat([x, xd], dim=1)
            
            x = self.conv_down5(down4)
            if describe: print(f'd5 x = {x.size()}') #1024
            xd = self.d_conv_down5(x)
            if describe: print(f'd5 xd = {xd.size()}') #512
            down5 = torch.cat([x, xd], dim=1)
            
            
            
            x = self.conv_up5(down5)
            if describe: print(f'u5 x = {x.size()}') #512
            xd = self.d_conv_up4(x)
            if describe: print(f'u4 xd = {xd.size()}') #256
            up4 = torch.cat([x, xd], dim=1)
            if describe: print(f'up4 = {up4.size()}') 
            
            x = self.conv_up4(up4)
            if describe: print(f'u4 x = {x.size()}') #256
            xs = torch.cat([x, down3], dim=1)
            if describe: print(f'u3 xs = {x.size()}') #256
            xd = self.d_conv_up3(xs)
            if describe: print(f'u3 xd = {xd.size()}') #128
            up3 = torch.cat([xs, xd], dim=1)
            if describe: print(f'up3 = {up3.size()}') 
            
            x = self.conv_up3(up3)
            if describe: print(f'u3 x = {x.size()}')
            xs = torch.cat([x, down2], dim=1)
            if describe: print(f'u2 xs = {x.size()}') #
            xd = self.d_conv_up2(xs)
            if describe: print(f'u2 xd = {xd.size()}') #512
            up2 = torch.cat([xs, xd], dim=1)
            if describe: print(f'up2 = {up2.size()}') 
            
            x = self.conv_up2(up2)
            if describe: print(f'u2 x = {x.size()}')
            xs = torch.cat([x, down1], dim=1)
            if describe: print(f'u1 xs = {x.size()}') #
            xd = self.d_conv_up1(xs)
            if describe: print(f'u1 xd = {xd.size()}') #512
            up1 = torch.cat([xs, xd], dim=1)
            if describe: print(f'up1 = {up1.size()}') 
            x = self.conv_last(up1) # 1
            if describe: print(f'conv_last = {x.size()}') 
            out = self.last(x)
            if describe: print(f'out = {out.size()}') 
            return out
            
            

    def get_weights_function(self, params):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])
                if params['conv_bias'] is not None:
                    m.bias.data.fill_(params['conv_bias'])
        return weights_init
 
