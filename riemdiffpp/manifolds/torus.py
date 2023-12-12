import torch
import numpy as np
from .base import Manifold

class Torus(Manifold):
    """
    Assumed to be scaled to [0, 2pi). Log value will be (-pi, pi].

    Also assumed to be of the form [B, D] where D is the dimension of the torus.
    """

    @classmethod
    def exp(cls, x, v):
        return (x + v).remainder(2 * np.pi)


    @classmethod
    def log(cls, x, y):
        v = (y - x).remainder(2 * np.pi)
        v[v > np.pi] -= 2 * np.pi
        return v
    

    @classmethod
    def expt(cls, x, v, t):
        return cls.exp(x, t[:, None] * v), v


    @classmethod
    def projx(cls, x):
        return x.remainder(2 * np.pi)


    @classmethod
    def proju(cls, x, v):
        return v


    @classmethod
    def sample_base(cls, batch_size, dim=2):
        return torch.rand(batch_size, dim) * (2 * np.pi)


    @classmethod
    def sample_base_like(cls, x):
        return cls.sample_base(x.shape[:-1], dim=x.shape[-1]).to(x)


    @classmethod
    def base_logp(cls, x):
        return -x.shape[-1] * np.log(2 * np.pi) * torch.ones_like(x[..., 0])


    @classmethod
    def _score_hk_wrapped(cls, x, x_orig, sigma, n=5):
        shifts = torch.arange(-n, n+1).to(x.device)
        dists = ((x - x_orig)[None, :] + 2 * np.pi * shifts[:, None, None]) / sigma[None, :, None]
        num_val = -(-0.5 * dists.pow(2)).exp() * dists / sigma[None, :, None]
        den_val = (-0.5 * dists.pow(2)).exp()
        return num_val.sum(dim=0) / den_val.sum(dim=0)
    

    @classmethod
    def _score_hk_ef(cls, x, x_orig, sigma, n=10):
        k = torch.arange(1, n + 1).to(x.device)

        kx = k[:, None, None] * x.unsqueeze(0)
        kx_orig = k[:, None, None] * x_orig.unsqueeze(0)

        kx_cos, kx_sin = kx.cos(), kx.sin()
        kx_orig_cos, kx_orig_sin = kx_orig.cos(), kx_orig.sin()

        const_factor = (-k.pow(2)[:, None] * sigma[None, :].pow(2) / 2).exp() / np.pi
        num_val = (k[:, None, None] * const_factor[..., None] * (kx_cos * kx_orig_sin - kx_sin * kx_orig_cos)).sum(dim=0)
        denom_val = 1 / (2 * np.pi) + (const_factor[..., None] * (kx_cos * kx_orig_cos + kx_sin * kx_orig_sin)).sum(dim=0)
        return num_val / denom_val


    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        # hyperparameters for optimal and fast approximation
        cutoff = 0.7
        n_ef = 8
        n_wrap = 5

        wrap_cond = sigma < cutoff
        ef_cond = torch.logical_not(wrap_cond)

        wrap_score = cls._score_hk_wrapped(x[wrap_cond], x_orig[wrap_cond], sigma[wrap_cond], n=n_wrap)
        ef_score = cls._score_hk_ef(x[ef_cond], x_orig[ef_cond], sigma[ef_cond], n=n_ef)

        score = torch.empty_like(x)
        score[wrap_cond] = wrap_score
        score[ef_cond] = ef_score

        return score

    @classmethod
    def sample_hk(cls, x_orig, sigma):
        return cls.projx(x_orig + sigma[:, None] * torch.randn_like(x_orig))
