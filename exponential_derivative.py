import jax
from jax import numpy as jnp, jacfwd
import numpy as np

from sh_representation import Lx, Ly, Lz, sh4_canonical

from icecream import ic


def commutativity(X, Y):
    return jnp.linalg.norm(X @ Y - Y @ X)


if __name__ == '__main__':
    np.random.seed(0)
    rotvec = np.random.randn(3)

    # Case 1
    g = lambda x: x * Lx
    f = lambda x: jax.scipy.linalg.expm(g(x)) @ sh4_canonical

    dX = jacfwd(g)(rotvec[0])
    X = g(rotvec[0])
    ic(commutativity(dX, X))

    # Since X, dX commute, we have \frac{df}{dx} = Lx \cdot f
    jac_auto = jacfwd(f)(rotvec[0])
    jac_analytical = Lx @ f(rotvec[0])
    ic(jnp.linalg.norm(jac_auto - jac_analytical))

    # The rest only hold for [0, 0, 0], i.e. multiplication with zero matrix always commute
    rotvec = np.array([0., 0., 0.])

    # Case 2 (4.1 equation 3)
    g = lambda v: v[0] * Lx + v[1] * Ly + v[2] * Lz
    dX = jacfwd(g)(rotvec)[..., 0]
    X = g(rotvec)
    ic(commutativity(dX, X))

    # For partial derivative \frac{\partial f}{\partial x} = Lx \cdot f no longer hold
    jac_auto = jacfwd(f)(rotvec)[..., 0]
    jac_analytical = Lx @ f(rotvec)
    ic(jnp.linalg.norm(jac_auto - jac_analytical))

    # Case 3 (Supplementary proof 1)
    ic(commutativity(rotvec[0] * Lx, rotvec[1] * Ly))

    # exp(X)exp(Y) = exp(X + Y) does not hold if X, Y do not commute
    sh_src = jax.scipy.linalg.expm(rotvec[0] * Lx +
                                   rotvec[1] * Ly) @ sh4_canonical
    sh_tar = jax.scipy.linalg.expm(rotvec[0] * Lx) @ jax.scipy.linalg.expm(
        rotvec[1] * Ly) @ sh4_canonical
    ic(jnp.linalg.norm(sh_src - sh_tar))
