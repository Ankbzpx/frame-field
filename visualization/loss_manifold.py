from common import (
    color_map,
    fibonacci_sphere,
    normalize,
    triangulation,
    write_triangle_mesh_VC,
)
from loss import (
    align_sh4_explicit,
    align_sh4_explicit_cosine,
    align_sh4_explicit_l2,
    align_sh4_functional,
)
from sh_representation import sh4_canonical

from jax import jit, numpy as jnp, vmap
import numpy as np
from tqdm import tqdm

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    randn_size = 100000
    sample_dirs = fibonacci_sphere(randn_size)
    sample_sh4s = jnp.repeat(sh4_canonical[None, ...], randn_size, axis=0)

    loss_tags = ["l1", "l2", "cosine_similarity", "functional"]
    loss_funcs = [
        align_sh4_explicit,
        align_sh4_explicit_l2,
        align_sh4_explicit_cosine,
        align_sh4_functional,
    ]

    colors = (
        jnp.array(
            [
                [255, 255, 255],
                [174, 216, 204],
                [205, 102, 136],
                [122, 49, 111],
                [70, 25, 89],
            ]
        )
        / 255.0
    )

    @jit
    def color_interp(val):
        idx = jnp.int32(val // 0.25)
        t = val % 0.25 / 0.25

        c0 = colors[idx]
        c1 = colors[idx + 1]

        return (1 - t) * c0 + t * c1

    for tag, func in zip(loss_tags, loss_funcs):
        print(tag)
        loss = func(sample_sh4s, sample_dirs)
        factor = (loss - loss.min()) / (loss.max() - loss.min())

        V = sample_dirs * (1 + factor[:, None])
        V, F = triangulation(V, sample_dirs)
        loss = func(
            jnp.repeat(sh4_canonical[None, ...], len(V), axis=0), vmap(normalize)(V)
        )
        factor = (loss - loss.min()) / (loss.max() - loss.min())
        VC = vmap(color_interp)(factor)

        write_triangle_mesh_VC(f"tmp/{tag}.obj", V, F, VC)
