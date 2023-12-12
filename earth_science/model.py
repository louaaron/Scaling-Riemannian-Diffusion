import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sph_harm import harmonics as sph_harmonics


class LearnedSiLU(nn.Module):
    def __init__(self, slope=1):
        super().__init__()
        self.slope = torch.nn.Parameter(slope * torch.ones(1))

    def forward(self, x):
        return self.slope * x * torch.sigmoid(x)


class SinActivation(torch.nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return torch.sin(x)


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

def create_network(in_dim, hidden_dim, out_dim, n_hidden, act):
    if act.lower() == "sin":
        act = SinActivation
    elif act.lower() == "silu":
        act = nn.SiLU
    else:
        act = LearnedSiLU

    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    network = [nn.Linear(in_dim, hidden_dim)]
    for _ in range(n_hidden - 1):
        network += [act(), nn.Linear(hidden_dim, hidden_dim)]
    network += [act(), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*network)


class Model(nn.Module):
    def __init__(self, cfg):
        # in_dim, hidden_dim, out_dim, n_hidden, act=nn.SiLU, t_steps=5, harm=False, harm_start=4, harm_end=6):
        super().__init__()
        self.t_steps = cfg.model.t_steps
        self.harm_start = cfg.model.harm_start
        self.harm_end = cfg.model.harm_end

        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.scale_by_sigma = cfg.model.scale_by_sigma

        self.diff = cfg.diff

        self.func = create_network(3 + 1 + cfg.model.t_steps, cfg.model.hidden_dim, 3, cfg.model.n_hidden, cfg.model.act)


    def forward(self, x, t):
        # turn t to sigma
        if self.diff:
            t = self.sigma_min ** (1 - t) * self.sigma_max ** t

        if not torch.is_tensor(t):
            t_p = t * torch.ones(x.shape[0]).to(x.device)
            t_p = t_p[:, None]
        else:
            t_p = t[:, None]

        inp = (x, t_p)
        if self.t_steps > 0:
            t_harmonics = get_timestep_embedding(t, self.t_steps)
            inp = inp + (t_harmonics,)

        input = torch.cat(inp, dim=-1)
        output = self.func(input)
        if self.scale_by_sigma and self.diff:
            return output / t_p
        else:
            return output

