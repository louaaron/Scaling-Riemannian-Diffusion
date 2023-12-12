import torch
import numpy as np

from torchdiffeq import odeint

def get_likelihood_fn(manifold, sampling_eps=1e-3, sm_ode=True, exact_div=False):

    @torch.no_grad()
    def likelihood_fn(drift_fn, data):
        eps = torch.randint_like(data, low=0, high=2).float() * 2 - 1.

        def ode_func(t, state):

            with torch.enable_grad():
                x, _ = state
                x.requires_grad_(True)
                if not torch.is_tensor(t) or t.numel() == 1:
                    t = t * torch.ones(x.shape[0]).to(x)
                drift = manifold.proju(x, drift_fn(x, t))
                if exact_div:
                    div = 0
                    for i in range(x.shape[-1]):
                        div -= torch.autograd.grad(drift[..., i].sum(), x, retain_graph= i < x.shape[-1] - 1)[0][..., i]
                else:
                    div = - (torch.autograd.grad((drift * eps).sum(), x)[0] * eps).sum(dim=-1)
            drift.detach_()
            x.requires_grad_(False)
            return drift, div

        state0 = (data, torch.zeros(data.shape[0]).to(data))
        if sm_ode:
            T = torch.tensor([sampling_eps, 1]).to(data)
        else:
            T = torch.tensor([1 - sampling_eps, 0]).to(data)
        base, dlogp = odeint(ode_func, state0, T, atol=1e-5, rtol=1e-5)
        base = base[1]
        dlogp = dlogp[1]
        base_logp = manifold.base_logp(base)

        return base_logp - dlogp

    return likelihood_fn