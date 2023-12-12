import torch
import numpy as np
from ..noise import schedule

def euler(func, x, t, manifold, steps=1000):
    time_steps = np.linspace(t[0], t[1], num=steps + 1)
    dt = 1 / steps * (-1 if t[0] > t[1] else 1)

    for t in time_steps[:-1]:
        t = t * torch.ones(x.shape[0]).to(x)
        x = manifold.exp(x, dt * func(x, t))
    return x
