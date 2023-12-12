import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Iterable

from math import pi

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels
    

def get_timestep_embedding(timesteps, embedding_dim, dtype=torch.float32):
    assert len(timesteps.shape) == 1
    timesteps *= 1000.

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = (torch.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb).exp()
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')


class Conv2d(nn.Module):
    def __init__(self,
        in_channels, out_channels, kernel, bias=True, pad_mode="circular", init_mode='kaiming_normal', init_weight=1, init_bias=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        init_kwargs = dict(mode=init_mode, fan_in=in_channels*kernel*kernel, fan_out=out_channels*kernel*kernel)

        self.weight = torch.nn.Parameter(weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs) * init_weight) if kernel else None
        self.bias = torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0

        if w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad)

        if b is not None:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x
        

class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = x.add_(self.bias.to(x.dtype))
        return x


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q.to(torch.float32), (k / np.sqrt(k.shape[1])).to(torch.float32)).softmax(dim=2).to(q.dtype)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw.to(torch.float32), output=w.to(torch.float32), dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k.to(torch.float32), db).to(q.dtype) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q.to(torch.float32), db).to(k.dtype) / np.sqrt(k.shape[1])
        return dq, dk



class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.conv1 = Conv2d(in_ch, out_ch, 3)
        self.conv2 = Conv2d(out_ch, out_ch, 3, init_weight=0)

        self.norm1 = nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6)
        self.norm2 = nn.GroupNorm(num_groups=min(out_ch // 4, 32), num_channels=out_ch, eps=1e-6)
        if in_ch != out_ch:
            self.skip = Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()
        
        self.cond_map = Linear(cond_dim, out_ch, bias=False, init_weight=0)

        self.dropout = dropout

    def forward(self, x, cond):
        h = x
        # activation for the last block
        h = F.silu(self.norm1(x))
        h = self.conv1(h)

        # add in conditioning
        h += self.cond_map(cond)[:, :, None, None]

        h = F.silu(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.conv2(h)
        x = h + self.skip(x)

        return x


class AttnBlock(nn.Module):
    """Self-attention residual block."""
    def __init__(self, channels, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6)
        self.qkv = Conv2d(channels, 3 * channels, 1)
        self.proj_out = Conv2d(channels, channels, 1, init_weight=0)

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
        w = AttentionOp.apply(q, k)
        a = torch.einsum('nqk,nck->ncq', w, v)
        x = self.proj_out(a.reshape(*x.shape)).add_(x)

        return x


class Model(nn.Module):
    def __init__(self, num_blocks, channels, dropout, attention, sigma_min, sigma_max):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels
        self.attention = attention
        input_ch = 18

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.cond_map = nn.Sequential(
            Linear(channels, 4 * channels),
            nn.SiLU(),
            Linear(4 * channels, 4 * channels),
        )
        
        self.conv_in = Conv2d(input_ch, channels, 3)

        
        # "downsampling"
        enc = []
        for _ in range(self.num_blocks):
            enc.append(ResNetBlock(channels, channels, 4 * channels, dropout=dropout))
            if self.attention:
                enc.append(AttnBlock(channels))
        self.enc = nn.ModuleList(enc)

        # middle
        self.mid1 = ResNetBlock(channels, channels, 4 * channels, dropout=dropout)
        self.midattn = AttnBlock(channels)
        self.mid2 = ResNetBlock(channels, channels, 4 * channels, dropout=dropout)

        # "upsampling"
        dec = []
        for _ in range(self.num_blocks + 1):
            dec.append(ResNetBlock(2 * channels, channels, 4 * channels, dropout=dropout))
            if self.attention:
                dec.append(AttnBlock(channels))
        self.dec = nn.ModuleList(dec)

        #  output
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=min(channels // 4, 32), num_channels=channels, eps=1e-6),
            nn.SiLU(),
            Conv2d(channels, input_ch, 3, init_weight=0)
        )

    def forward(self, x, cond):
        # complex to real
        # assume x is of the form [B, 16, 16, 3, 3]
        x = torch.view_as_real(x)
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 18)
        x = x.permute(0, 3, 1, 2)

        # normalize to sigma
        cond = self.sigma_min ** (1 - cond) * self.sigma_max ** cond

        sigma_inp = cond
        temb = get_timestep_embedding(cond, self.channels)
        cond = self.cond_map(temb)

        outputs = []

        x = self.conv_in(x)
        outputs.append(x)

        for i in range(self.num_blocks):
            if self.attention:
                x = self.enc[2 * i](x, cond)
                x = self.enc[2 * i + 1](x)
            else:
                x = self.enc[i](x, cond)
            outputs.append(x)
        
        x = self.mid1(x, cond)
        x = self.midattn(x)
        x = self.mid2(x, cond)

        for i in range(self.num_blocks + 1):   
            if self.attention:
                x = self.dec[2 * i](torch.cat((x, outputs.pop()), dim=1), cond)
                x = self.dec[2 * i + 1](x)
            else:
                x = self.dec[i](torch.cat((x, outputs.pop()), dim=1), cond)
        
        if len(outputs) > 0:
            raise ValueError("Something went wrong with the blocks")

        out = self.out(x)

        out = out / sigma_inp[:, None, None, None]

        # output to complex [B, 16, 16, 3, 3]
        out = out.permute(0, 2, 3, 1)
        out = out.reshape(out.shape[0], out.shape[1], out.shape[2], 3, 3, 2).contiguous()
        out = torch.view_as_complex(out)
        
        return out