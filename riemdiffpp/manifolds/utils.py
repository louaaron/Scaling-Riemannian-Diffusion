import torch


class Sindiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x.sin() / x
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x.abs() < 1e-6, y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (x * x.cos() - x.sin()) / x.pow(2)
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < 1e-6, y_stable, y) * g


sindiv = Sindiv.apply


class Divsin(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        y = x / x.sin()
        y_stable = torch.ones_like(x)
        ctx.save_for_backward(x)
        return torch.where(x.abs() < 1e-6, y_stable, y)

    @staticmethod
    def backward(ctx, g):
        x, = ctx.saved_tensors
        y = (1 - x * x.cos() / x.sin()) / x.sin()
        y_stable = torch.zeros_like(x)
        return torch.where(x.abs() < 1e-6, y_stable, y) * g


divsin = Divsin.apply


class Sinkx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k):
        x, _ = ctx.saved_tensors
        y = (k * x).sin() / x.sin()
        y_stable = k * torch.ones_like(x)
        

    @staticmethod
    def backward(ctx, g):
        pass

sinkx = Sinkx.apply


def unsqueeze_as(x, y, back=True):
    """
    Unsqueeze x to have as many dimensions as y. For example, tensor shapes:

    x: (a, b, c), y: (a, b, c, d, e) -> output: (a, b, c, 1, 1)
    """
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)