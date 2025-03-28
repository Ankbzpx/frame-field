from model_jax import LipMLP, Siren

import jax
from jax import jit, numpy as jnp, vmap
import matplotlib.pyplot as plt
import numpy as np

from icecream import ic


if __name__ == "__main__":
    np.random.seed(0)

    N = 10000
    x = np.random.uniform(-1, 1, (N, 3))
    input_scale = 1

    key = jax.random.PRNGKey(0)
    # mlp = Siren(3, 256, 4, 1, key, input_scale=input_scale)
    mlp = LipMLP(
        3,
        256,
        4,
        1,
        key,
        activation="sin",
        input_scale=input_scale,
    )

    samples = [x]

    x = mlp.input_scale * x
    for i in range(len(mlp.layers)):
        x = vmap(mlp.layers[i])(x)
        samples.append(x)
        if i != len(mlp.layers) - 1:
            x = getattr(jax.nn, mlp.activation)(x)
            samples.append(x)

    num_plots = len(samples)
    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 20))

    for i, ax in enumerate(axes):
        ax.hist(samples[i].reshape(-1), bins=30, alpha=0.7, color=f"C{i}")

    plt.tight_layout()
    plt.show()
