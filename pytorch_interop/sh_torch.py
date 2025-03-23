import numpy as np
import torch

from icecream import ic


def r_2(x, y, z):
    return x**2 + y**2 + z**2


def r_4(x, y, z):
    return r_2(x, y, z) ** 2


def y_00(x, y, z):
    return (1 / 2) * np.sqrt(1 / np.pi) * r_4(x, y, z)


def y_4_4(x, y, z):
    return (3 / 4) * np.sqrt(35 / np.pi) * x * y * (x**2 - y**2)


def y_4_3(x, y, z):
    return (3 / 4) * np.sqrt(35 / (2 * np.pi)) * y * (3 * x**2 - y**2) * z


def y_4_2(x, y, z):
    return (3 / 4) * np.sqrt(5 / np.pi) * x * y * (7 * z**2 - r_2(x, y, z))


def y_4_1(x, y, z):
    return (3 / 4) * np.sqrt(5 / (2 * np.pi)) * y * (7 * z**3 - 3 * z * r_2(x, y, z))


def y_40(x, y, z):
    return (
        (3 / 16)
        * np.sqrt(1 / np.pi)
        * (35 * z**4 - 30 * z**2 * r_2(x, y, z) + 3 * r_4(x, y, z))
    )


def y_41(x, y, z):
    return (3 / 4) * np.sqrt(5 / (2 * np.pi)) * x * (7 * z**3 - 3 * z * r_2(x, y, z))


def y_42(x, y, z):
    return (3 / 8) * np.sqrt(5 / np.pi) * (x**2 - y**2) * (7 * z**2 - r_2(x, y, z))


def y_43(x, y, z):
    return (3 / 4) * np.sqrt(35 / (2 * np.pi)) * x * (x**2 - 3 * y**2) * z


def y_44(x, y, z):
    return (
        (3 / 16)
        * np.sqrt(35 / np.pi)
        * (x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2))
    )


zonal_z_poly_scale = (3 * 35) / (16 * np.sqrt(np.pi))
zonal_z_00 = 21 / 8
zonal_z_20 = 3 * np.sqrt(5) / 2
zonal_z_40 = 1

# oct_poly_scale * (x^4 + y^4 + z^4) = oct_00 * y_00 + sqrt(7 / 12) * y_40 + sqrt(5 / 12) * y_44
oct_00 = 3 * np.sqrt(21) / 4
# r^4 is NOT in denominator because it has been **pre-multiplied** to basis
oct_poly_scale = 5 * np.sqrt(21 / np.pi) / 8
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
    return np.sqrt(4 * np.pi / (2 * l + 1))


def zonal_sh4_coeffs(u):
    return torch.hstack([zonal_band_coeff(4) * zonal_z_40 * eval_sh4_basis(u)])


def R3_to_sh4_zonal(R3):
    return zonal_to_octa_scale * zonal_sh4_coeffs(torch.transpose(R3, -1, -2)).sum(1)


if __name__ == "__main__":
    x = torch.randn(100, 3, 3)
    ic(R3_to_sh4_zonal(x).shape)
