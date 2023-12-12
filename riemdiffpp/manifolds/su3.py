import torch
import numpy as np
from .base import Manifold


from .utils import sindiv, divsin, unsqueeze_as

class SU3(Manifold):

    @classmethod
    def exp(cls, x, v):
        return x @ torch.matrix_exp(x.adjoint() @ v)

    @classmethod
    def exp_la(cls, x, v):
        # assume that v is in the lie algebra
        return x @ torch.matrix_exp(v)

    @classmethod
    def log(cls, x, y, move_back=True):
        raise ValueError("Not Needed")
        
    @classmethod
    def dist(cls, x, y):
        raise ValueError("Not Needed")
    
    @classmethod
    def expt(cls, x, v, t):
        raise ValueError("Not Implemented")

    @classmethod
    def projx(cls, x):
        U, _, Vh = torch.linalg.svd(x, full_matrices=False)
        val = U @ Vh
        return val / val.det()[..., None, None].pow(1/3)

    @classmethod
    def proju_la(cls, v):
        B = (v - v.adjoint()) / 2
        trace = torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
        B = B - 1/3 * trace * unsqueeze_as(torch.eye(3, device=v.device), trace, back=False)
        return B

    @classmethod
    def proju(cls, x, v):
        v = x.adjoint() @ v
        B = (v - v.adjoint()) / 2
        trace = torch.diagonal(B, dim1=-2, dim2=-1).sum(dim=-1)[..., None, None]
        B = B - 1/3 * trace * unsqueeze_as(torch.eye(3, device=x.device), trace, back=False)
        return x @ B

    @classmethod
    def sample_base(cls, batch_size, dim=None):
        z = torch.randn(batch_size, 3, 3, dtype=torch.cfloat)
        q, r = torch.linalg.qr(z)
        d = torch.diagonal(r, dim1=-2, dim2=-1)
        ph = d / d.abs()
        q = q @ torch.diag_embed(ph) @ q
        q = q / torch.det(q).pow(1/3)[..., None, None]
        return q

    @classmethod
    def sample_base_like(cls, x):
        return cls.sample_base(x.shape[0], dim=None).to(x)
    
    @classmethod
    def base_logp(cls, x):
        return -(0.5 * np.log(3) + 5 * np.log(np.pi)) * torch.ones(x.shape[0])

    @classmethod
    def _hk_sop(cls, x, x_orig, sigma, n=15):
        M = x_orig.adjoint() @ x
        e_vals = torch.linalg.eigvals(M)
        thetas = torch.angle(e_vals[..., :2])
        A, B = thetas[..., 0], thetas[..., 1]
        Al = A[..., None] + 2 * np.pi * unsqueeze_as(torch.arange(-n, n+1).to(A), A[..., None], back=False)
        Bm = B[..., None] + 2 * np.pi * unsqueeze_as(torch.arange(-n, n+1).to(B), B[..., None], back=False)

        Al = Al[..., None, :]
        Bm = Bm[..., :, None]

        vdet = divsin((Al - Bm) / 2) * divsin(Al / 2 + Bm) * divsin(Al + Bm / 2)
        e_pow = (-(Al.pow(2) + Al * Bm + Bm.pow(2)) / unsqueeze_as(sigma.pow(2), vdet)).exp()
        sum_term = (vdet * e_pow).flatten(-2).sum(dim=-1)
        fac = unsqueeze_as(16 * (sigma.pow(2)).exp() / (2 * np.pi * sigma.pow(2)).pow(4), sum_term)
        return fac * sum_term

    @classmethod
    def _score_hk_sop(cls, x, x_orig, sigma, n=4):
        with torch.enable_grad():
            x.requires_grad_(True)

            val = cls._hk_sop(x, x_orig, sigma, n=n)
            grad = torch.autograd.grad(val.sum(), x)[0]

            x.requires_grad_(False)
            val = val.detach()
            grad = grad.detach()

            grad = cls.proju(x, grad)

        return grad / val[..., None, None]
    
    @classmethod
    def _hk_ef(cls, x, x_orig, sigma, n=10):
        # not implemented because it's too inefficient on GPU. Formulas are given in
        # https://iopscience.iop.org/article/10.1088/0305-4470/21/11/022/pdf
        # https://en.wikipedia.org/wiki/Clebsch%E2%80%93Gordan_coefficients_for_SU(3)
        pass
    
    @classmethod
    def hk(cls, x, x_orig, sigma):
        raise ValueError("Not Implemented")
        
    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        return cls._score_hk_sop(x, x_orig, sigma)

    @classmethod
    def _sample_hk_bf(cls, x_orig, sigma, steps=200):
        # resort to brute force sampling
        x = x_orig
        t = sigma.pow(2) / 2

        with torch.no_grad():
            for i in range(steps):
                x = cls.exp_la(x, cls.proju_la(unsqueeze_as((t / steps).sqrt(), x) * torch.randn_like(x)))
        return cls.projx(x)
    
    @classmethod
    def sample_hk(cls, x_orig, sigma):
        return cls._sample_hk_bf(x_orig, sigma, steps=100)