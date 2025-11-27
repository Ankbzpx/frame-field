import math

from common_pytorch import normalize
import numpy as np
import torch

from icecream import ic


sh4_canonical = torch.tensor(
    [0, 0, 0, 0, math.sqrt(7 / 12), 0, 0, 0, math.sqrt(5 / 12)]
)


def r_2(x, y, z):
    return x**2 + y**2 + z**2


def r_4(x, y, z):
    return r_2(x, y, z) ** 2


def y_00(x, y, z):
    return (1 / 2) * math.sqrt(1 / math.pi) * r_4(x, y, z)


def y_4_4(x, y, z):
    return (3 / 4) * math.sqrt(35 / math.pi) * x * y * (x**2 - y**2)


def y_4_3(x, y, z):
    return (3 / 4) * math.sqrt(35 / (2 * math.pi)) * y * (3 * x**2 - y**2) * z


def y_4_2(x, y, z):
    return (3 / 4) * math.sqrt(5 / math.pi) * x * y * (7 * z**2 - r_2(x, y, z))


def y_4_1(x, y, z):
    return (
        (3 / 4) * math.sqrt(5 / (2 * math.pi)) * y * (7 * z**3 - 3 * z * r_2(x, y, z))
    )


def y_40(x, y, z):
    return (
        (3 / 16)
        * math.sqrt(1 / math.pi)
        * (35 * z**4 - 30 * z**2 * r_2(x, y, z) + 3 * r_4(x, y, z))
    )


def y_41(x, y, z):
    return (
        (3 / 4) * math.sqrt(5 / (2 * math.pi)) * x * (7 * z**3 - 3 * z * r_2(x, y, z))
    )


def y_42(x, y, z):
    return (3 / 8) * math.sqrt(5 / math.pi) * (x**2 - y**2) * (7 * z**2 - r_2(x, y, z))


def y_43(x, y, z):
    return (3 / 4) * math.sqrt(35 / (2 * math.pi)) * x * (x**2 - 3 * y**2) * z


def y_44(x, y, z):
    return (
        (3 / 16)
        * math.sqrt(35 / math.pi)
        * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
    )


zonal_z_poly_scale = (3 * 35) / (16 * math.sqrt(math.pi))
zonal_z_00 = 21 / 8
zonal_z_20 = 3 * math.sqrt(5) / 2
zonal_z_40 = 1

# oct_poly_scale * (x^4 + y^4 + z^4) = oct_00 * y_00 + sqrt(7 / 12) * y_40 + sqrt(5 / 12) * y_44
oct_00 = 3 * math.sqrt(21) / 4
# r^4 is NOT in denominator because it has been **pre-multiplied** to basis
oct_poly_scale = 5 * math.sqrt(21 / math.pi) / 8
zonal_to_octa_scale = oct_poly_scale / zonal_z_poly_scale

sh4_basis = [y_4_4, y_4_3, y_4_2, y_4_1, y_40, y_41, y_42, y_43, y_44]


def eval_sh4_basis(v):
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    return torch.stack(
        [
            y_4_4(x, y, z),
            y_4_3(x, y, z),
            y_4_2(x, y, z),
            y_4_1(x, y, z),
            y_40(x, y, z),
            y_41(x, y, z),
            y_42(x, y, z),
            y_43(x, y, z),
            y_44(x, y, z),
        ],
        dim=-1,
    )


def zonal_band_coeff(l):
    return math.sqrt(4 * math.pi / (2 * l + 1))


def zonal_sh4_coeffs(u):
    return torch.hstack([zonal_band_coeff(4) * zonal_z_40 * eval_sh4_basis(u)])


def R3_to_sh4_zonal(R3):
    return zonal_to_octa_scale * zonal_sh4_coeffs(torch.transpose(R3, -1, -2)).sum(1)


sh0_basis = [y_00]
sh4_basis = [y_4_4, y_4_3, y_4_2, y_4_1, y_40, y_41, y_42, y_43, y_44]


def eval_oct_basis(v):
    x = v[0]
    y = v[1]
    z = v[2]
    return torch.tensor([f(x, y, z) for f in sh0_basis + sh4_basis])


def oct_polynomial_sh4(v, sh4):
    sh = torch.hstack([torch.tensor(oct_00, device=v.device), sh4])
    return torch.dot(sh, eval_oct_basis(v) / oct_poly_scale)


def grad_oct_polynomial_sh4(v, sh4):
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]

    x2 = x * x
    y2 = y * y
    z2 = z * z

    x3 = x * x * x
    y3 = y * y * y
    z3 = z * z * z

    coeffs = torch.tensor(
        [
            1.0 / 2.0 * math.sqrt(1.0 / math.pi),
            3.0 / 4.0 * math.sqrt(35.0 / math.pi),
            3.0 / 4.0 * math.sqrt(35.0 / 2 / math.pi),
            3.0 / 4.0 * math.sqrt(5.0 / math.pi),
            3.0 / 4.0 * math.sqrt(5.0 / 2.0 / math.pi),
            3.0 / 16.0 * math.sqrt(1.0 / math.pi),
            3.0 / 4.0 * math.sqrt(5.0 / 2.0 / math.pi),
            3.0 / 8.0 * math.sqrt(5.0 / math.pi),
            3.0 / 4.0 * math.sqrt(35.0 / 2 / math.pi),
            3.0 / 16.0 * math.sqrt(35.0 / math.pi),
        ],
        device=x.device,
    )

    dx = torch.stack(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * x,
            3.0 * x2 * y - y3,
            6.0 * x * y * z,
            6.0 * y * z2 - 3.0 * x2 * y - y3,
            -6.0 * x * y * z,
            -60.0 * x * z2 + 6.0 * (x2 + y2 + z2) * 2.0 * x,
            4.0 * z3 - 9.0 * x2 * z - 3.0 * y2 * z,
            12.0 * x * z2 - 4.0 * x3,
            3.0 * x2 * z - 3.0 * y2 * z,
            4.0 * x3 - 12.0 * x * y2,
        ],
        dim=-1,
    )

    dy = torch.stack(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * y,
            x3 - 3.0 * x * y2,
            3.0 * x2 * z - 3.0 * y2 * z,
            6.0 * x * z2 - x3 - 3.0 * x * y2,
            4.0 * z3 - 3.0 * x2 * z - 9.0 * y2 * z,
            -60.0 * y * z2 + 6.0 * (x2 + y2 + z2) * 2.0 * y,
            -6.0 * x * y * z,
            -12.0 * y * z2 + 4.0 * y3,
            -6.0 * x * y * z,
            -12.0 * x2 * y + 4.0 * y3,
        ],
        dim=-1,
    )

    dz = torch.stack(
        [
            2.0 * (x2 + y2 + z2) * 2.0 * z,
            torch.zeros_like(x),
            3.0 * x2 * y - y3,
            12.0 * x * y * z,
            12.0 * y * z2 - 3.0 * x2 * y - 3.0 * y3,
            20.0 * z3 - 60.0 * x2 * z - 60.0 * y2 * z + 6.0 * (x2 + y2 + z2) * 2.0 * z,
            12.0 * x * z2 - 3.0 * x3 - 3.0 * x * y2,
            12.0 * x2 * z - 12.0 * y2 * z,
            x3 - 3.0 * x * y2,
            torch.zeros_like(x),
        ],
        dim=-1,
    )
    sh = torch.hstack([(oct_00 * torch.ones_like(x))[..., None], sh4])
    return (
        torch.stack(
            [
                (coeffs[None, :] * dx * sh).sum(-1),
                (coeffs[None, :] * dy * sh).sum(-1),
                (coeffs[None, :] * dz * sh).sum(-1),
            ],
            dim=-1,
        )
        / oct_poly_scale
    )


def proj_sh4_to_R3(sh4s_target: torch.Tensor, max_iter=1000):
    if len(sh4s_target.shape) < 2:
        sh4s_target = sh4s_target[None, ...]

    # Needs to be normalized
    sh4s_target = normalize(sh4s_target)

    n_elem = len(sh4s_target)

    torch.random.manual_seed(0)
    v1 = torch.randn((n_elem, 3), device=sh4s_target.device)
    v2 = torch.randn((n_elem, 3), device=sh4s_target.device)

    # sqrt(n_elem * eps**2)
    min_loss = math.sqrt(n_elem) * 1e-8

    loss = 100.0
    iter = 0

    while loss > min_loss and iter < max_iter:
        # Power iteration
        v1_ = grad_oct_polynomial_sh4(v1, sh4s_target)
        v1_ = normalize(v1_)
        v2_ = grad_oct_polynomial_sh4(v2, sh4s_target)
        v2_ = v2_ - (v1_ * v2_).sum(-1)[:, None] * v1_
        v2_ = normalize(v2_)

        loss = torch.linalg.matrix_norm(v1 - v1_)
        v1 = v1_
        v2 = v2_
        iter += 1

    v3 = torch.cross(v1, v2, dim=-1)
    return torch.stack([v1, v2, v3], -1)


def align_sh4_functional_grad(sh4, normal):
    normal = normalize(normal)
    grad_normal = grad_oct_polynomial_sh4(normal, normalize(sh4))
    return torch.linalg.norm(4 * normal - grad_normal, ord=2, dim=-1)
