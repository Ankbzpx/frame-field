from sh_pytorch import grad_oct_polynomial_sh4
import torch


def normalize(x):
    return x / (torch.linalg.norm(x, dim=-1, keepdim=True) + 1e-8)


def align_sh4_functional_grad(sh4, normal):
    grad_normal = grad_oct_polynomial_sh4(normalize(normal), normalize(sh4))
    return torch.linalg.norm(4 * normal - grad_normal, ord=2, dim=-1)


def eikonal(x):
    return torch.abs(torch.linalg.norm(x, dim=-1) - 1)
