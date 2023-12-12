import torch
import numpy as np

def get_sm_loss(manifold, sigma_min=0.001, sigma_max=4, eps=1e-5, sliced=False):
    def loss_fn(model, data):
        t = torch.rand(data.shape[0]).to(data.device) * (1 - eps) + eps
        sigma = sigma_min * (sigma_max / sigma_min) ** t
        x = manifold.sample_hk(data, sigma)

        if sliced:
            epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
            with torch.enable_grad():
                x.requires_grad_(True)
                score = manifold.proju(x, model(x, t))
                div = (torch.autograd.grad((score * epsilon).sum(), x, retain_graph=True)[0] * epsilon).sum(dim=-1)

            div.detach()
            x.requires_grad_(False)
            loss = sigma.pow(2) * (2 * div + score.norm(dim=-1).pow(2))
        else:
            score_hk = manifold.score_hk(x, data, sigma)
            score = manifold.proju(x, model(x, t))

            # print(score_hk.isnan().any(), score.isnan().any())
            loss = sigma.pow(2) * (score - score_hk).flatten(1).norm(dim=-1).pow(2)
        return loss
    
    return loss_fn


def get_prob_ode(manifold, model, sigma_min=0.001, sigma_max=4):
    def drift_fn(x, t):
        x = manifold.projx(x)

        if not torch.is_tensor(t) or t.numel() == 1:
            t = t * torch.ones(x.shape[0]).to(x.device)
        sigma = sigma_min * (sigma_max / sigma_min) ** t
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=t.device))

        score = manifold.proju(x, model(x, t))
        return -diffusion[:, None] ** 2 * score / 2
    return drift_fn