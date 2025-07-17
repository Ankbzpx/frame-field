from common import normalize_aabb, vis_oct_field
from loss import align_sh4_functional_grad
from model_jax import Siren
from sh_representation import proj_sh4_to_R3

import equinox as eqx
import igl
import jax
from jax import grad, jit, numpy as jnp, vmap
import numpy as np
import optax
from tqdm import tqdm
import trimesh

from icecream import ic
import polyscope as ps


if __name__ == "__main__":
    np.random.seed(0)

    V, F = igl.read_triangle_mesh("data/toy/30.obj")

    crease = trimesh.Trimesh(V, F)
    VN = np.array(crease.face_normals)
    samples, fids = trimesh.sample.sample_surface_even(crease, 10000)
    sample_normals = VN[fids]
    latent = jnp.empty((len(samples), 0))

    viz_size = 16
    axis = jnp.linspace(-0.25, 0.25, viz_size)
    xx, yy = jnp.meshgrid(axis, axis)
    xx -= 0.15
    yy += 0.15
    vis_samples = jnp.stack([xx, jnp.zeros_like(xx), yy], axis=-1).reshape(-1, 3)
    vis_latent = jnp.empty((len(vis_samples), 0))

    key = jax.random.PRNGKey(0)
    model = Siren(3, 256, 4, 9, key, final_activation="normalize")

    lr = 5e-5
    optim = optax.adam(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    align_weight = 50
    relative_weight = 0.01
    smooth_weight = relative_weight * align_weight

    @eqx.filter_jit
    @eqx.filter_grad(has_aux=True)
    def loss_func(model, samples, sample_normals):
        jac, sh4 = model.call_jac(samples, latent)
        loss_align = (
            align_weight * align_sh4_functional_grad(sh4, sample_normals).mean()
        )
        loss_smooth = (
            smooth_weight * vmap(jnp.linalg.norm, in_axes=(0, None))(jac, "f").mean()
        )
        loss = loss_align + loss_smooth
        return loss, loss

    @eqx.filter_jit
    def make_step(model, opt_state, samples, sample_normals):
        grads, loss = loss_func(model, samples, sample_normals)
        updates, opt_state = optim.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n_iters = 1000
    pbar = tqdm(range(n_iters), dynamic_ncols=True)
    for iter in pbar:
        model, opt_state, loss = make_step(model, opt_state, samples, sample_normals)
        pbar.set_postfix({"loss": loss})

    sh4 = model(vis_samples, vis_latent)
    R3 = proj_sh4_to_R3(sh4)
    V_vis, F_vis = vis_oct_field(R3, vis_samples, 0.01)
    V[:, 1] *= 0.1

    # sh4 = model(samples, latent)
    # R3 = proj_sh4_to_R3(sh4)
    # V_vis, F_vis = vis_oct_field(R3, samples, 0.01)

    ps.init()
    ps.register_surface_mesh("ms", V, F)
    ps.register_surface_mesh("Vis", V_vis, F_vis)
    # ps.register_point_cloud("Samples", samples).add_vector_quantity(
    #     "Sample_normals", sample_normals, enabled=True
    # )
    ps.show()
