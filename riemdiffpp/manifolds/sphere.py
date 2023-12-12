import torch
import numpy as np
from scipy.special import loggamma
from .base import Manifold


from .utils import sindiv, divsin

class Sphere(Manifold):
    # assume dim=3 because otherwise the sampling must be significantly different.

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
    def sample_base(cls, batch_size, dim=2):
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
    def _hk_ef(cls, x, x_orig, sigma, n=50):
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

        return ((2 * ns + dim - 2) / (dim - 2) * geg).sum(dim=0) / np.exp(cls._logv(dim))
    
    @classmethod
    def _log_hk_ef(cls, x, x_orig, sigma, n=100):
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
        st_cond = theta < 1e-6
        gen_cond = torch.logical_not(st_cond)
        score[st_cond] = y_stable[st_cond]
        score[gen_cond] = y[gen_cond]

        return score


    @classmethod
    def _hk_wrap(cls, x, x_orig, sigma):
        dim = x.shape[-1]
        theta = (x * x_orig).sum(dim=-1).clip(min=-1, max=1).acos()
        return sindiv(theta).pow(dim - 2) / (2 * np.pi * sigma ** 2).pow((dim - 1) / 2) / (theta ** 2 / 2 / sigma ** 2).exp()
    
    
    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        # hyperparameters
        cutoff_wrap = 0.4

        cond_ef = sigma > cutoff_wrap
        cond_wrap = torch.logical_not(cond_ef)

        score_ef = cls._score_hk_ef(x[cond_ef], x_orig[cond_ef], sigma[cond_ef], n=10)
        score_wrap = cls._score_hk_wkb(x[cond_wrap], x_orig[cond_wrap], sigma[cond_wrap])

        score = torch.empty_like(x)
        score[cond_ef] = score_ef
        score[cond_wrap] = score_wrap
        return score

    @classmethod
    def _M_unif(cls, sigma, n=20):
        inn = torch.ones_like(sigma)
        leg = [torch.ones_like(inn), inn]

        for i in range(2, n + 1):
            leg.append(((2 * i - 1) * inn * leg[-1] - (i - 1) * leg[-2]) / i)

        leg = torch.stack(leg, dim=0)
        ns = torch.arange(0, n + 1).to(inn)[:, None]
        return ((2 * ns + 1) * (- ns * (ns + 1) * sigma[None, :].pow(2) / 2).exp() * leg).sum(dim=0)

    @classmethod
    def _sample_reject_unif(cls, x_orig, sigma, n=10):
        samples = torch.empty_like(x_orig)
        sampled = torch.zeros_like(sigma).to(torch.bool)
        M = cls._M_unif(sigma) * 1.01

        i = 0

        while not sampled.all():
            i += 1
            index = torch.arange(0, sigma.shape[0]).to(sigma.device)[torch.logical_not(sampled)]
            x_orig_s = x_orig[torch.logical_not(sampled)]
            sigma_s = sigma[torch.logical_not(sampled)]

            new_samples = cls.sample_base_like(x_orig_s)
            p_unif = 1 / (4 * np.pi)
            p_hk = cls._hk_ef(new_samples, x_orig_s, sigma_s, n=n)

            u = torch.rand_like(sigma_s)
            accept = u < p_hk / (M[index] * p_unif)

            sampled[index[accept]] = True
            samples.data[index[accept]] = new_samples[accept]
        # print(i)
        return samples

    @classmethod
    def _M_wrap(cls, sigma):
        cutoffs = torch.tensor([1.05, 1.1, 1.2, 1.3, 1.65, 1.75, 2.25]).to(sigma) # < 0.05 to < 0.35
        return cutoffs[torch.floor(sigma * 20).clamp(min=0, max=6).int()]

    @classmethod
    def _sample_reject_wrap(cls, x_orig, sigma, n=100):
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
            p_wrap = cls._hk_wrap(new_samples, x_orig_s, sigma_s)
            p_hk = cls._hk_ef(new_samples, x_orig_s, sigma_s, n=n)

            u = torch.rand_like(sigma_s)
            accept = u < p_hk / (M[index] * p_wrap)
            sampled[index[accept]] = True
            samples.data[index[accept]] = new_samples[accept]
        # print(i)
        return samples

    @classmethod
    def _sample_wrap(cls, x_orig, sigma):
        v = cls.proju(x_orig, torch.randn_like(x_orig) * sigma[:, None])
        return cls.exp(x_orig, v)

    @classmethod
    def sample_hk(cls, x_orig, sigma):
        cutoff_wrap = 0.04
        cutoff_wrap_reject = 0.35

        evals_wrap = 100
        evals_unif = 10

        cond_wrap = sigma < cutoff_wrap
        cond_wrap_reject = torch.logical_and(sigma >= cutoff_wrap, sigma < cutoff_wrap_reject)
        cond_unif_reject = sigma >= cutoff_wrap_reject
        

        samples_wrap = cls._sample_wrap(x_orig[cond_wrap], sigma[cond_wrap])
        samples_wr = cls._sample_reject_wrap(x_orig[cond_wrap_reject], sigma[cond_wrap_reject], n=evals_wrap)
        samples_ur = cls._sample_reject_unif(x_orig[cond_unif_reject], sigma[cond_unif_reject], n=evals_unif)

        samples = torch.empty_like(x_orig)
        samples[cond_wrap] = samples_wrap
        samples[cond_wrap_reject] = samples_wr
        samples[cond_unif_reject] = samples_ur

        return samples

    @classmethod
    def _hk_mt(cls, theta, sigma):
        # normalized probability measure (by angle theta). Used for testing.
        inn = theta.cos()
        dim = 3
        n = 5000
        alpha = dim / 2 - 1

        fac = sigma ** 2 / 2
        geg = [torch.ones_like(inn), 2 * alpha * inn * ((1 - dim) * fac).exp()]

        for i in range(2, n + 1):
            new_geg = (2 * (i + alpha - 1) * inn * geg[-1] - ((5 - 2 * i - dim) * fac).exp() * (i + 2 * alpha - 2) * geg[-2]) / i
            new_geg *= ((3 - 2 * i - dim) * fac).exp()
            geg.append(new_geg)

        geg = torch.stack(geg, dim=0)
        ns = torch.arange(0, n + 1).to(inn)[:, None]

        return ((2 * ns + dim - 2) / (dim - 2) * geg).sum(dim=0) * theta.sin() / 2