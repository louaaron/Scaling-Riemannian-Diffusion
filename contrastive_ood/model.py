import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode='kaiming_normal', init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = torch.nn.Parameter(weight_init([out_features, in_features], **init_kwargs) * init_weight)
        self.bias = torch.nn.Parameter(weight_init([out_features], **init_kwargs) * init_bias) if bias else None

    def forward(self, x):
        x = x @ self.weight.t()
        if self.bias is not None:
            x = x + self.bias
        return x


class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim, dropout=0.1):
        super().__init__()
        self.lin1 = Linear(in_ch, out_ch)
        self.lin2 = Linear(out_ch, out_ch)

        self.norm1 = nn.LayerNorm(in_ch)
        self.norm2 = nn.LayerNorm(out_ch)
        if in_ch != out_ch:
            self.skip = Linear(in_ch, out_ch)
        else:
            self.skip = nn.Identity()
        
        self.cond_map = Linear(cond_dim, out_ch, bias=False, init_weight=0)


        self.dropout = dropout

    def forward(self, x, cond):
        h = x
        # activation for the last block
        h = F.silu(self.norm1(x))
        h = self.lin1(h)

        # add in conditioning
        h += self.cond_map(cond)

        h = F.silu(self.norm2(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = self.lin2(h)
        x = h + self.skip(x)

        return x


class AttentionOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k):
        w = torch.einsum('ncq,nck->nqk', q, k / np.sqrt(k.shape[1])).softmax(dim=2)
        ctx.save_for_backward(q, k, w)
        return w

    @staticmethod
    def backward(ctx, dw):
        q, k, w = ctx.saved_tensors
        db = torch._softmax_backward_data(grad_output=dw, output=w, dim=2, input_dtype=torch.float32)
        dq = torch.einsum('nck,nqk->ncq', k, db) / np.sqrt(k.shape[1])
        dk = torch.einsum('ncq,nqk->nck', q, db) / np.sqrt(k.shape[1])
        return dq, dk


class AttnBlock(nn.Module):
    """Self-attention residual block."""
    def __init__(self, dim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = Linear(dim, 3 * dim)
        self.proj_out = Linear(dim, dim, init_weight=0)

    def forward(self, x):
        q, k, v = self.qkv(self.norm(x)).reshape(x.shape[0] * self.num_heads, x.shape[1] // self.num_heads, 3, -1).unbind(2)
        w = AttentionOp.apply(q, k)
        a = torch.einsum('nqk,nck->ncq', w, v)
        x = self.proj_out(a.reshape(*x.shape)).add_(x)

        return x


class LDM(nn.Module):
    def __init__(self, input_dim, dim, num_blocks, dropout, sigma_min, sigma_max, precond):
        super().__init__()

        self.dim = dim
        self.cond_map = nn.Sequential(
            Linear(dim, 4 * dim),
            nn.SiLU(),
            Linear(4 * dim, 4 * dim),
        )

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.precond = precond

        self.lin_in = Linear(input_dim, dim)

        
        # downsampling
        enc = []
        for _ in range(num_blocks):
            enc.append(ResNetBlock(dim, dim, 4 * dim, dropout=dropout))
        self.enc = nn.ModuleList(enc)

        # middle
        self.mid1 = ResNetBlock(dim, dim, 4 * dim, dropout=dropout)
        self.midattn = AttnBlock(dim)
        self.mid2 = ResNetBlock(dim, dim, 4 * dim, dropout=dropout)

        # "upsampling"
        dec = []
        for _ in range(num_blocks + 1):
            dec.append(ResNetBlock(2 * dim, dim, 4 * dim, dropout=dropout))
        self.dec = nn.ModuleList(dec)

        #  output
        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.SiLU(),
            Linear(dim, input_dim, init_weight=0)
        )

    def forward(self, x, cond):
        if self.precond:
            cond = self.sigma_min ** (1 - cond) * self.sigma_max ** cond
        sigma_inp = cond
        t = cond
        temb = get_timestep_embedding(t, self.dim)
        cond = self.cond_map(temb)

        outputs = []

        x = self.lin_in(x)
        outputs.append(x)

        for block in self.enc:
            x = block(x, cond)
            outputs.append(x)

        
        x = self.mid1(x, cond)
        x = self.midattn(x)
        x = self.mid2(x, cond)

        for block in self.dec:
            x = block(torch.cat((x, outputs.pop()), dim=1), cond)
        
        if len(outputs) > 0:
            raise ValueError("Something went wrong with the blocks")

        out = self.out(x)

        out = out / sigma_inp[:, None]
        
        return out
