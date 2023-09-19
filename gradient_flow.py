import numpy as np
import jax
from jax import numpy as jnp, vmap, jit, grad
from common import normalize
from sh_representation import oct_polynomial_sh4, oct_polynomial_sh4_unit_norm, oct_polynomial_zonal, oct_polynomial_zonal_unit_norm, oct_polynomial, sh4_canonical
from loss import double_well_potential

from icecream import ic
import polyscope as ps


@jit
def decompose(n, v):
    v_para = jnp.dot(v, n) * n
    return v_para, v - v_para


def vis_gradient(VN, gradient, local_minimums, scale=1e-2):
    gradient_para, gradient_orth = vmap(decompose)(VN, gradient)
    grad_vis_len, grad_para_vis_len, grad_orth_vis_len = jax.tree_map(
        lambda x: scale * len(VN) * jnp.linalg.norm(x, axis=1).max(),
        [gradient, gradient_para, gradient_orth])

    ps.init()
    pc = ps.register_point_cloud('pc', VN, radius=1e-4)
    pc.add_vector_quantity('gradient',
                           gradient,
                           enabled=True,
                           length=grad_vis_len)
    pc.add_vector_quantity('gradient_para',
                           gradient_para,
                           length=grad_para_vis_len)
    pc.add_vector_quantity('gradient_orth',
                           gradient_orth,
                           length=grad_orth_vis_len)
    ps.register_point_cloud('local_minimums', local_minimums)
    ps.show()


if __name__ == '__main__':
    # Without loss of generality, we choose canonical basis
    sh4 = sh4_canonical
    R = jnp.eye(3)

    local_minimums = np.hstack([R, -R]).T

    # The goal of our loss is to push VN to one of the basis, up to the change of sign
    VN = vmap(normalize)(np.random.randn(100000, 3))

    @jit
    def loss_basis(VN, basis):
        dps = jnp.einsum('ij,bi->bj', basis, vmap(normalize)(VN))
        return double_well_potential(jnp.abs(dps)).sum(-1).mean()

    @jit
    def loss_sh4(VN, sh4):
        return vmap(oct_polynomial_sh4, in_axes=[0, None])(VN, sh4).mean()

    @jit
    def loss_sh4_unit_norm(VN, sh4):
        return vmap(oct_polynomial_sh4_unit_norm,
                    in_axes=[0, None])(VN, sh4).mean()

    @jit
    def loss_poly(VN, R):
        return vmap(oct_polynomial, in_axes=[0, None])(VN, R).mean()

    @jit
    def loss_zonal(VN, R):
        return vmap(oct_polynomial_zonal, in_axes=[0, None])(VN, R).mean()

    @jit
    def loss_zonal_unit_norm(VN, R):
        return vmap(oct_polynomial_zonal_unit_norm,
                    in_axes=[0, None])(VN, R).mean()

    # Note the gradient of both `loss_basis` and `loss_sh4_unit_norm` lies on the sphere, but with opposite source and sink
    # gradient = -grad(loss_basis)(VN, R)
    # gradient = grad(loss_sh4)(VN, sh4)
    # gradient = grad(loss_sh4_unit_norm)(VN, sh4)
    # gradient = grad(loss_poly)(VN, R)
    # gradient = grad(loss_zonal)(VN, R)
    gradient = grad(loss_zonal_unit_norm)(VN, R)

    vis_gradient(VN, gradient, local_minimums)