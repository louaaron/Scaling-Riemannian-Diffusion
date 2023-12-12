import torch
import numpy as np
from .base import Manifold


from .utils import sindiv, divsin

class SO3(Manifold):

    @classmethod
    def exp(cls, x, v):
        v_base = x.transpose(-1, -2) @ v

        # use rodrigues formula, which is numerically stable
        theta = v_base[..., [0, 0, 1], [1, 2, 2]].norm(dim=-1).unsqueeze(-1).unsqueeze(-1)
        k = v_base / theta
        r = torch.matrix_power(k, 0) + theta.sin() * k + (1 - theta.cos()) * torch.matrix_power(k, 2)

        return x @ r

    @classmethod
    def log(cls, x, y, move_back=True):
        r = x.transpose(-1, -2) @ y
        val = (((r[..., range(3), range(3)]).sum(dim=-1) - 1) / 2).clip(min=-1, max=1)
        theta = val.acos()
        log_val = divsin(theta)[..., None, None] / 2 * (r - r.transpose(-1, -2))

        if move_back:
            return x @ log_val
        else:
            return log_val
        
    @classmethod
    def dist(cls, x, y):
        r = x.transpose(-1, -2) @ y
        val = (((r[..., range(3), range(3)]).sum(dim=-1) - 1) / 2).clip(min=-1, max=1)
        theta = val.acos()
        return theta
    
    @classmethod
    def expt(cls, x, v, t):
        vt_base = t[..., None, None] * x.transpose(-1, -2) @ v

        theta = vt_base[..., [0, 0, 1], [1, 2, 2]].norm(dim=-1)
        k = vt_base / theta
        r = torch.matrix_power(k, 0) + theta.sin() * k + (1 - theta.cos()) * torch.matrix_power(k, 2)

        # x exp(t x^{-1} v) vs v exp(t x^{-1} v)
        return x @ r, v @ r

    @classmethod
    def projx(cls, x):
        """
        Use svd nearest projection.
        """
        U, _, Vh = torch.linalg.svd(x, full_matrices=False)
        return U @ Vh

    @classmethod
    def proju(cls, x, v):
        return (v - x @ v.transpose(-1, -2) @ x) / 2

    @classmethod
    def sample_base(cls, batch_size, dim=None):
        quat = torch.randn(batch_size, 4)
        quat /= quat.norm(dim=-1, keepdim=True)

        x, y, z, w = quat.transpose(-1, -2)

        Q = torch.stack([
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - w * y), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
        ], dim=-1).reshape(batch_size, 3, 3)

        return Q

    @classmethod
    def sample_base_like(cls, x):
        return cls.sample_base(x.shape[0], dim=None).to(x)
    
    @classmethod
    def base_logp(cls, x):
        return -np.log(8 * np.pi ** 2) * torch.ones_like(x[..., 0, 0])
    
    @classmethod
    def _hk_ef(cls, x, x_orig, sigma, n=10):
        # never used
        theta = cls.dist(x_orig, x)
        l = torch.arange(0, n+1).to(x)[:, None]
        summa = (2 * l + 1) * (-l * (l + 1) * sigma[None, :].pow(2) / 2).exp() * ((2 * l + 1) * theta / 2).sin() / (theta / 2).sin()
        return summa.sum(dim=0) / (8 * np.pi ** 2)

    @classmethod
    def _score_hk_ef(cls, x, x_orig, sigma, n=3):
        with torch.enable_grad():
            x.requires_grad_(True)

            val = cls._hk_ef(x, x_orig, sigma, n=n)
            grad = torch.autograd.grad(val.sum(), x)[0]

            x.requires_grad_(False)
            val = val.detach()
            grad = grad.detach()

            grad = cls.proju(x, grad)

        return grad / val[..., None, None]


    @classmethod
    def _hk_wrap(cls, x, x_orig, sigma, n=7):
        theta = cls.dist(x_orig, x)
        n_vals = torch.arange(-n, n+1).to(x)[:, None] * 2 * np.pi
        theta_shifted = theta[None, :] + n_vals
        # print(theta_shifted[:, 0])
        summa = divsin(theta_shifted / 2) * (-theta_shifted.pow(2) / (2 * sigma[None, :].pow(2))).exp()
        return summa.sum(dim=0) * (sigma.pow(2) / 8).exp() / np.power(2 * np.pi, 1.5) / sigma.pow(3)

    @classmethod
    def _score_hk_wrap(cls, x, x_orig, sigma, n=7):
        with torch.enable_grad():
            x.requires_grad_(True)

            val = cls._hk_wrap(x, x_orig, sigma, n=n)
            grad = torch.autograd.grad(val.sum(), x)[0]

            x.requires_grad_(False)
            val = val.detach()
            grad = grad.detach()

            grad = cls.proju(x, grad)

        return grad / val[..., None, None]
    
    @classmethod
    def _hk_vard(cls, x, x_orig, sigma):
        return (-cls.dist(x, x_orig).pow(2) / sigma.pow(2) / 2).exp() / np.power(2 * np.pi, 1.5) / sigma.pow(3)
    
    @classmethod
    def _score_hk_vard(cls, x, x_orig, sigma):
        return cls.log(x, x_orig) / 2 / sigma[:, None, None].pow(2)
    
    @classmethod
    def hk(cls, x, x_orig, sigma):
        # include cutoff due to numerical instability near origin
        cond_vard = cls.dist(x, x_orig) < 1e-4
        cond_wrap = torch.logical_not(cond_vard)

        x_vard = x[cond_vard]
        x_o_vard = x_orig[cond_vard]
        sigma_vard = sigma[cond_vard]
        hk_vard = cls._hk_vard(x_vard, x_o_vard, sigma_vard)

        x_wrap = x[cond_wrap]
        x_o_wrap = x_orig[cond_wrap]
        sigma_wrap = sigma[cond_wrap]
        hk_wrap = cls._hk_wrap(x_wrap, x_o_wrap, sigma_wrap)

        hk = torch.empty_like(sigma)
        hk[cond_vard] = hk_vard
        hk[cond_wrap] = hk_wrap
        return hk
        
    
    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        cutoff_vard = 0.01
        cutoff_wrap = 1.7
        cond_vard = sigma < cutoff_vard
        cond_wrap = torch.logical_and(sigma >= cutoff_vard, sigma < cutoff_wrap)
        cond_ef = sigma >= cutoff_wrap

        score_hk_vard = cls._score_hk_vard(x[cond_vard], x_orig[cond_vard], sigma[cond_vard])
        score_hk_wrap = cls._score_hk_wrap(x[cond_wrap], x_orig[cond_wrap], sigma[cond_wrap])
        score_hk_ef = cls._score_hk_ef(x[cond_ef], x_orig[cond_ef], sigma[cond_ef])

        score_hk = torch.empty_like(x_orig)
        score_hk[cond_vard] = score_hk_vard
        score_hk[cond_wrap] = score_hk_wrap
        score_hk[cond_ef] = score_hk_ef
        
        return score_hk

    @classmethod
    def _wrap_p(cls, x, x_orig, sigma, n=6):
        theta = cls.dist(x_orig, x)
        n_vals = torch.arange(-n, n+1).to(x)[:, None] * 2 * np.pi
        theta_shifted = theta[None, :] + n_vals
        summa = sindiv(theta_shifted / 2).pow(2) * (-theta_shifted.pow(2) / (2 * sigma[None, :].pow(2))).exp()
        return summa.sum(dim=0) / np.power(2 * np.pi, 1.5) / sigma.pow(3)
    
    @classmethod
    def _sample_wrap(cls, x_orig, sigma):
        v = cls.proju(x_orig, torch.randn_like(x_orig) * sigma[:, None, None])
        return cls.exp(x_orig, v)
    
    @classmethod
    def _M_wrap(cls, sigma):
        cutoffs = torch.tensor([1.05, 1.18, 1.4, 1.8, 2.4, 3.4, 4.15, 4.2, 4.3, 4.42, 4.54]).to(sigma) # < 0.1 to < 1.1
        return cutoffs[torch.floor(sigma * 10).clamp(min=0, max=10).int()]

    @classmethod
    def _sample_reject_wrap(cls, x_orig, sigma):
        samples = torch.empty_like(x_orig)
        sampled = torch.zeros_like(sigma).to(torch.bool)
        M = cls._M_wrap(sigma)

        i = 0
        while not sampled.all():
            i += 1
            index = torch.arange(0, sigma.shape[0]).to(sigma.device)[torch.logical_not(sampled)]
            x_orig_s = x_orig[torch.logical_not(sampled)]
            sigma_s = sigma[torch.logical_not(sampled)]

            new_samples = cls._sample_wrap(x_orig_s, sigma_s)
            p_wrap = cls._wrap_p(new_samples, x_orig_s, sigma_s)
            p_hk = cls.hk(new_samples, x_orig_s, sigma_s)

            u = torch.rand_like(sigma_s)
            accept = u < p_hk / (M[index] * p_wrap)
            # print(p_hk / (M[index] * p_wrap).max())
            sampled[index[accept]] = True
            samples.data[index[accept]] = new_samples[accept]
        # print(i)
        return samples
    
    @classmethod
    def _M_unif(cls, sigma):
        l = torch.arange(0, 15).to(sigma)[:, None]
        summa = (2 * l + 1) ** 2 * (-l * (l + 1) * sigma[None, :].pow(2) / 2).exp()
        return summa.sum(dim=0) * 1.1 # extra factor for numerical stability

    @classmethod
    def _sample_reject_unif(cls, x_orig, sigma):
        samples = torch.empty_like(x_orig)
        sampled = torch.zeros_like(sigma).to(torch.bool)
        M = cls._M_unif(sigma)

        i = 0

        while not sampled.all():
            i += 1
            index = torch.arange(0, sigma.shape[0]).to(sigma.device)[torch.logical_not(sampled)]
            x_orig_s = x_orig[torch.logical_not(sampled)]
            sigma_s = sigma[torch.logical_not(sampled)]

            new_samples = cls.sample_base_like(x_orig_s)
            p_unif = 1 / (8 * np.pi ** 2)
            p_hk = cls.hk(new_samples, x_orig_s, sigma_s)

            u = torch.rand_like(sigma_s)
            accept = u < p_hk / (M[index] * p_unif)

            sampled[index[accept]] = True
            samples.data[index[accept]] = new_samples[accept]
        # print(i)
        return samples
    
    @classmethod
    def sample_hk(cls, x_orig, sigma):
        cutoff_wrap = 1e-3
        cutoff_wrap_reject = 1.1

        cond_wrap = sigma < cutoff_wrap
        cond_wrap_reject = torch.logical_and(sigma >= cutoff_wrap, sigma < cutoff_wrap_reject)
        cond_unif_reject = sigma >= cutoff_wrap_reject

        samples_wrap = cls._sample_wrap(x_orig[cond_wrap], sigma[cond_wrap])
        samples_wr = cls._sample_reject_wrap(x_orig[cond_wrap_reject], sigma[cond_wrap_reject])
        samples_ur = cls._sample_reject_unif(x_orig[cond_unif_reject], sigma[cond_unif_reject])

        samples = torch.empty_like(x_orig)
        samples[cond_wrap] = samples_wrap
        samples[cond_wrap_reject] = samples_wr
        samples[cond_unif_reject] = samples_ur

        return samples
    
    @classmethod
    def _hk_mt(cls, theta, sigma):
        # normalized probability measure (by angle theta). Used for testing.
        l = torch.arange(0, 21).to(theta)[:, None]
        summa = (2 * l + 1) * (-l * (l + 1) * sigma[None, :].pow(2) / 2).exp() * ((2 * l + 1) * theta / 2).sin() / (theta / 2).sin()
        return (1 - theta.cos()) / np.pi * summa.sum(dim=0)