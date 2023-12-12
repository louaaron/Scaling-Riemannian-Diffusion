import torch
from tqdm import tqdm
import numpy as np
from time import time
from torch.func import grad_and_value

from riemdiffpp import SU3

step_size = 1e-3
steps = 4000
batch = 20000
L = 2
N = 10


def potential(grid, beta=9):
    """
    Link is complex value of size B, 16, 16, 3, 3
    """
    l = list(range(1, grid.shape[1])) + [0]
    linke = grid
    linkn = grid[:, l]
    linkw = grid[:, l][:, :, l]
    links = grid[:, :, l]

    p_trace = torch.diagonal(linke @ linkn @ linkw.adjoint() @ links.adjoint(), dim1=-2, dim2=-1).sum(dim=-1)
    pot = torch.real(p_trace).flatten(1).sum(dim=-1) * beta / 3
    return pot.sum()

samples = SU3.sample_base(L ** 2 * batch).reshape(batch, L, L, 3, 3).cuda()

for i in range(steps):
    g, v = grad_and_value(potential)(samples)
    print(i, v / batch)
    # if v / 10000 > 8.2:
    #     break
    g = SU3.proju(samples, step_size * g / 2 + np.sqrt(step_size) * torch.randn_like(samples))
    samples = SU3.exp(samples, g)
    if i % 100 == 0 and i > 0:
        samples = SU3.projx(samples)

    if samples.isnan().any():
        print(f"Nan at step {i}.")
        break
    
print("Saved")
torch.save(samples, f"/atlas2/u/aaronlou/datasets/su3_qcd_{L}.pth")
