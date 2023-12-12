import torch
import numpy as np
from scipy.special import loggamma
from .base import Manifold


from .utils import sindiv, divsin

class HyperSphere(Manifold):
    # assume dim=128

    @classmethod
    def exp(cls, x, v):
        norm_v = v.norm(dim=-1, keepdim=True)
        return x * norm_v.cos() + v * sindiv(norm_v)

    @classmethod
    def log(cls, x, y):
        xy = (x * y).sum(dim=-1, keepdim=True)
        xy.data.clamp_(min=-1, max=1)
        val = torch.acos(xy)
        return divsin(val) * (y - xy * x)

    @classmethod
    def dist(cls, x, y):
        return (x * y).sum(dim=-1).clip(min=-1, max=1).acos()
    
    @classmethod
    def expt(cls, x, v, t):
        t = t[:, None]
        norm_v = v.norm(dim=-1, keepdim=True)
        
        y = x * (t * norm_v).cos() + t * v * sindiv(t * norm_v)
        deriv = - x * (t * norm_v).sin() * norm_v + v * (t * norm_v).cos()
        return y, deriv

    @classmethod
    def projx(cls, x):
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    @classmethod
    def proju(cls, x, v):
        return v - x * (x * v).sum(dim=-1, keepdim=True)

    @classmethod
    def sample_base(cls, batch_size, dim=127):
        x = torch.randn(batch_size, dim+1)
        x /= x.norm(dim=-1, keepdim=True)
        return x

    @classmethod
    def sample_base_like(cls, x):
        return cls.sample_base(x.shape[0], dim=x.shape[-1] - 1).to(x)

    @classmethod
    def _logv(cls, dim):
        # note that dim is (d + 1) where d is manifold dim
        return np.log(2) + dim / 2 * np.log(np.pi) - loggamma(dim / 2)
    
    @classmethod
    def base_logp(cls, x):
        dim = x.shape[-1]
        return -cls._logv(dim) * torch.ones_like(x[..., 0])
    
    @classmethod
    def _log_hk_ef(cls, x, x_orig, sigma, n=100, norm=False):
        """
        Return log of the heat kernel (as given with eigenfunctions).
        """
        dim = x.shape[-1]
        inn = (x * x_orig).sum(dim=-1)
        alpha = dim / 2 - 1

        fac = sigma ** 2 / 2
        geg = [torch.ones_like(inn), 2 * alpha * inn * ((1 - dim) * fac).exp()]

        for i in range(2, n + 1):
            new_geg = (2 * (i + alpha - 1) * inn * geg[-1] - ((5 - 2 * i - dim) * fac).exp() * (i + 2 * alpha - 2) * geg[-2]) / i
            new_geg *= ((3 - 2 * i - dim) * fac).exp()
            geg.append(new_geg)

        geg = torch.stack(geg, dim=0)
        ns = torch.arange(0, n + 1).to(inn)[:, None]

        if norm:
            return ((2 * ns + dim - 2) / (dim - 2) * geg).sum(dim=0).log()
        else:
            return ((2 * ns + dim - 2) / (dim - 2) * geg).sum(dim=0).log() - cls._logv(dim)

    @classmethod
    def _score_hk_ef(cls, x, x_orig, sigma, n=50):

        with torch.enable_grad():
            x.requires_grad_(True)

            val = cls._log_hk_ef(x, x_orig, sigma, n=n)
            grad = torch.autograd.grad(val.sum(), x)[0]

            x.requires_grad_(False)
            grad = grad.detach()

            grad = cls.proju(x, grad)

        return grad

    @classmethod
    def _score_hk_wkb(cls, x, x_orig, sigma):
        dim = x.shape[-1]
        theta = (x * x_orig).sum(dim=-1).clip(min=-1, max=1).acos()
        theta_deriv = (dim/2 - 1) * (1/theta - theta.cos() / theta.sin()) - theta / sigma ** 2
        y = (theta_deriv / theta)[:, None] * - cls.log(x, x_orig)
        y_stable = torch.zeros_like(x)

        score = torch.empty_like(x)
        st_cond = theta < 1e-3
        gen_cond = torch.logical_not(st_cond)
        score[st_cond] = y_stable[st_cond]
        score[gen_cond] = y[gen_cond]

        return score
    
    
    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        # hyperparameters
        dim = x.shape[-1]
        assert dim in [128, 512]
        cutoff_wrap = 0.08 if dim == 128 else 0.044
        ef_dtype = torch.double if dim == 512 else torch.float
        n_hk = 100 if dim == 128 else 200

        cond_ef = sigma > cutoff_wrap
        cond_wrap = torch.logical_not(cond_ef)

        score_ef = cls._score_hk_ef(x[cond_ef].to(ef_dtype), x_orig[cond_ef].to(ef_dtype), sigma[cond_ef], n=n_hk).float()
        score_wrap = cls._score_hk_wkb(x[cond_wrap], x_orig[cond_wrap], sigma[cond_wrap])

        score = torch.empty_like(x)
        score[cond_ef] = score_ef
        score[cond_wrap] = score_wrap
        return score


    @classmethod
    def _sample_hk_bf(cls, x_orig, sigma, steps=100):
        # resort to brute force sampling since dynamics are too high dimensional
        x = x_orig
        t = sigma.pow(2) / 2

        with torch.no_grad():
            for i in range(steps):
                x = cls.exp(x, cls.proju(x, (t / steps).sqrt()[:, None] * torch.randn_like(x)))
        return x

    @classmethod
    def sample_hk(cls, x_orig, sigma, steps=100):
        return cls._sample_hk_bf(x_orig, sigma, steps=steps)
