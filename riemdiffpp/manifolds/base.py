import abc
import torch


class Manifold:

    @classmethod
    def inner(cls, x, u, v):
        return (u * v).sum(dim=-1, keepdim=True)

    @classmethod
    def exp(cls, x, v):
        pass

    @classmethod
    def log(cls, x, y):
        pass
    
    @classmethod
    def expt(cls, x, v, t):
        """
        Compute exp_x(tv) and the derivative with respect to t
        """

    @classmethod
    def projx(cls, x):
        pass

    @classmethod
    def proju(cls, x, v):
        pass

    @classmethod
    def sample_base(cls, batch_size, dim=None):
        pass

    @classmethod
    def sapmle_base_like(cls, x):
        pass
    
    @classmethod
    def base_logp(cls, x):
        pass
    
    @classmethod
    def score_hk(cls, x, x_orig, sigma):
        pass

    @classmethod
    def sample_hk(cls, x_orig, sigma):
        pass