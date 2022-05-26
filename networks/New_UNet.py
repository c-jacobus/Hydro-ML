import torch
import torch.nn as nn

import logging
from utils import logging_utils
logging_utils.config_logger()

from einops import rearrange

def down_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 4, stride=2, padding=1),
        nn.LeakyReLU(inplace=True),
    )   

def up_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, 4, stride=2, padding=1, output_padding=0),
        nn.ReLU(inplace=True),
    )


def inverse_transf(x):
    return torch.exp(14.*x)


def loss_func(gen_output, target, lambda_rho):

    # first part of the loss
    l1_loss = nn.functional.l1_loss(gen_output, target)

    if lambda_rho == 0.:
        return l1_loss

    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output[:,0,:,:,:])
    orig_tar = inverse_transf(target[:,0,:,:,:])
    orig_l1_loss = nn.functional.l1_loss(orig_gen, orig_tar)
    return l1_loss + lambda_rho * orig_l1_loss


@torch.jit.script
def loss_func_opt(gen_output: torch.Tensor, target: torch.Tensor, lambda_rho: float):

    # first part of the loss
    l1_loss = torch.mean(torch.abs(gen_output - target))

    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output[:,0,:,:,:])
    orig_tar = inverse_transf(target[:,0,:,:,:])
    orig_l1_loss = torch.mean(torch.abs(orig_gen - orig_tar))
    return l1_loss + lambda_rho * orig_l1_loss  


@torch.jit.script
def loss_func_opt_final(gen_output: torch.Tensor, target: torch.Tensor, lambda_rho: torch.Tensor):

    # first part of the loss
    l1_loss = torch.abs(gen_output - target)
    
    # Transform T and rho back to original space, compute additional L1
    orig_gen = inverse_transf(gen_output)
    orig_tar = inverse_transf(target)
    orig_l1_loss = torch.abs(orig_gen - orig_tar)

    # combine
    loss = l1_loss + lambda_rho * orig_l1_loss
    
    return torch.mean(loss)




# FROM Diffusion:

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv3d(dim, dim, 4, 2, 1)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# building block modules

class ConvNextBlock(nn.Module):
    """ https://arxiv.org/abs/2201.03545 """

    def __init__(self, dim, dim_out, *, time_emb_dim = None, mult = 2, norm = True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.GELU(),
            nn.Linear(time_emb_dim, dim)
        ) if exists(time_emb_dim) else None

        self.ds_conv = nn.Conv3d(dim, dim, 7, padding = 3, groups = dim)

        self.net = nn.Sequential(
            LayerNorm(dim) if norm else nn.Identity(),
            nn.Conv3d(dim, dim_out * mult, 3, padding = 1),
            nn.GELU(),
            nn.Conv3d(dim_out * mult, dim_out, 3, padding = 1)
        )

        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):
        h = self.ds_conv(x)

        if exists(self.mlp):
            assert exists(time_emb), 'time emb must be passed in'
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, 'b c -> b c 1 1')

        h = self.net(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        k = k.softmax(dim = -1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv3d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)


class UNet(nn.Module):
    
    def __init__(self, params):
        # logging.info("Initializing Model Parameters...")
        super().__init__()
        
        dim = 8
        dim_mults=(1, 2, 4, 8)
        channels = params.N_in_channels
        with_time_emb = False

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        time_dim = None
        self.time_mlp = None
        
        # logging.info("Initializing Model Layers...")
        # logging.info(f'Down Channels: {in_out}')

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ConvNextBlock(dim_in, dim_out, time_emb_dim = time_dim, norm = ind != 0),
                ConvNextBlock(dim_out, dim_out, time_emb_dim = time_dim),
                # Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        # self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ConvNextBlock(mid_dim, mid_dim, time_emb_dim = time_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):  # used to be [1:]
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ConvNextBlock(dim_out * 2, dim_in, time_emb_dim = time_dim),
                ConvNextBlock(dim_in, dim_in, time_emb_dim = time_dim),
                # Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))

        self.final_conv = nn.Sequential(
            ConvNextBlock(dim, dim),
            nn.Conv3d(dim, params.N_out_channels, 1)
        )

    def forward(self, x):
        #logging.info(f'Input Shape: {x.size()}')
        t =  None

        h = []

        for convnext, convnext2, downsample in self.downs: # convnext, convnext2, attn, downsample
            x = convnext(x, t)
            x = convnext2(x, t)
            # x = attn(x)
            h.append(x)
            x = downsample(x)
            #logging.info(f'(Down) Shape: {x.size()}')

        x = self.mid_block1(x, t)
        # x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        #logging.info(f'(Mid) Shape: {x.size()}')

        for convnext, convnext2, upsample in self.ups: # convnext, convnext2, attn, downsample
            x = torch.cat((x, h.pop()), dim=1)
            x = convnext(x, t)
            x = convnext2(x, t)
            # x = attn(x)
            x = upsample(x)
            #logging.info(f'(Up) Shape: {x.size()}')

        
        output = self.final_conv(x)
        #logging.info(f'Output Shape: {output.size()}')
        return output

    def get_weights_function(self, params):
        def weights_init(m):
            classname = m.__class__.__name__
            if type(m) == nn.Linear:       # if classname.find('Conv') != -1 and classname != 'ConvNextBlock':
                nn.init.normal_(m.weight.data, 0.0, params['conv_scale'])
                if params['conv_bias'] is not None:
                    m.bias.data.fill_(params['conv_bias'])
        return weights_init

