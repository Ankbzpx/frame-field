import matplotlib.pyplot as plt
import numpy as np
import optax
from jax import vmap

from icecream import ic

if __name__ == '__main__':
    total_steps = 20
    scheduler = optax.polynomial_schedule(1, 0, 4, total_steps, 0)

    steps = np.arange(total_steps)

    ic(scheduler(0.75 * total_steps))

    plt.figure()
    plt.plot(steps, vmap(scheduler)(steps))
    plt.savefig('test.png')
