import torch
from .noise import schedule
from .solvers import *


def sde_sampler(manifold, solver="euler"):
    """
    Euler-Maruyama sampler shouldn't be hard to implement.
    """
    raise NotImplementedError("Not implemented")


def ode_sampler(manifold, solver="euler", sampling_eps=1e-5, device=torch.device('cpu'), solver_kwargs={}):
    solver = SOLVER_DICT[solver]

    @torch.no_grad()
    def sampler(func, batch_size, dim=None):
        base = manifold.sample_base(batch_size, dim=dim).to(device)
        output = solver(func, base, (1, sampling_eps), manifold, **solver_kwargs)
        return output

    return sampler


def fm_ode_sampler(manifold, solver="euler", sampling_eps=1e-5, device=torch.device('cpu'), solver_kwargs={}):
    solver = SOLVER_DICT[solver]

    @torch.no_grad()
    def sampler(func, batch_size, dim=None):
        base = manifold.sample_base(batch_size, dim=dim).to(device)
        output = solver(func, base, (0, 1 - sampling_eps), manifold, **solver_kwargs)
        return output

    return sampler