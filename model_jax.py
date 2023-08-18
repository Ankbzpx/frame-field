import jax
from jax import vmap, numpy as jnp, jacfwd
import equinox as eqx
from jaxtyping import Array

from icecream import ic
from functools import partial


# https://github.com/google/jax/pull/762
def value_and_jacfwd(f, x):
    pushfwd = partial(jax.jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac


class Linear(eqx.Module):
    W: Array
    b: Array

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 xavier_init: bool = False):

        if xavier_init:
            self.W = jax.random.uniform(
                key, (out_features, in_features), minval=-1.,
                maxval=1.) * jnp.sqrt(6. / (in_features + out_features))
        else:
            self.W = jax.random.normal(
                key, (out_features, in_features)) * jnp.sqrt(2. / in_features)

        self.b = jnp.zeros(out_features)

    def __call__(self, x):
        return self.W @ x + self.b


class MLP(eqx.Module):

    def __init__():
        pass

    def single_call(self, x):
        pass

    def single_call_split(self, x):
        x = self.single_call(x)
        return x[0], x[1:]

    def call_grad(self, x):
        return vmap(
            eqx.filter_value_and_grad(self.single_call_split, has_aux=True))(x)

    def call_jac(self, x):
        return vmap(value_and_jacfwd, in_axes=[None, 0])(self.single_call, x)

    def __call__(self, x):
        x = vmap(self.single_call)(x)
        return x

    def get_aux_loss(self):
        return 0


class StandardMLP(MLP):
    layers: list[eqx.Module]
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu',
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation

        self.layers = [
            Linear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            Linear(hidden_features, hidden_features, keys[i + 1], xavier_init)
            for i in range(hidden_layers)
        ] + [Linear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)
        return x


class ResMLP(MLP):
    layers: list[eqx.Module]
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu',
                 **kwargs):

        keys = jax.random.split(key, 2 * hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation

        self.layers = [
            Linear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            Linear(hidden_features, hidden_features, keys[i + 1], xavier_init)
            for i in range(2 * hidden_layers)
        ] + [Linear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x):
        activation = getattr(jax.nn, self.activation)

        x = self.layers[0](x)
        x = activation(x)

        for i in range((len(self.layers) - 2) // 2):
            out = self.layers[2 * i + 1](x)
            out = activation(out)
            out = self.layers[2 * (i + 1)](out)
            x = activation(x + out)

        x = self.layers[-1](x)
        return x


# Modified from: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/model.py
class LipLinear(Linear):
    W: Array
    b: Array
    c: Array

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 xavier_init: bool = False):
        super().__init__(in_features, out_features, key, xavier_init)
        self.c = jnp.max(jnp.sum(jnp.abs(self.W), axis=1))

    # L-infinity weight normalization
    def weight_normalization(self, W, softplus_c):
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]

    def __call__(self, x):
        return self.weight_normalization(self.W, jax.nn.softplus(
            self.c)) @ x + self.b


class LipMLP(MLP):
    layers: list[eqx.Module]
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='tanh',
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation

        self.layers = [
            LipLinear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            LipLinear(hidden_features, hidden_features, keys[i + 1],
                      xavier_init) for i in range(hidden_layers)
        ] + [LipLinear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)
        return x

    # Lipschitz loss
    # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/7048aada5db1e6603a3d13fb1bc1ee2c61762985/demos/lipschitz_mlp/model.py#L82
    def get_aux_loss(self):
        loss_lip = 1.0
        for layer in self.layers:
            loss_lip = loss_lip * jax.nn.softplus(layer.c)
        return loss_lip


class SineLayer(eqx.Module):
    W: Array
    b: Array
    omega_0: float

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 is_first: bool = False,
                 omega_0: float = 30):
        self.omega_0 = omega_0

        if is_first:
            self.W = jax.random.uniform(key, (out_features, in_features),
                                        minval=-1.,
                                        maxval=1.) / in_features
        else:
            self.W = jax.random.uniform(
                key, (out_features, in_features), minval=-1.,
                maxval=1.) * jnp.sqrt(6 / in_features) / omega_0

        self.b = jnp.zeros(out_features)

    def __call__(self, x):
        return jnp.sin(self.omega_0 * (self.W @ x + self.b))


class Siren(MLP):
    layers: list[eqx.Module]

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 first_omega_0: float = 30,
                 hidden_omega_0: float = 30,
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)

        self.layers = [
            SineLayer(in_features,
                      hidden_features,
                      keys[0],
                      is_first=True,
                      omega_0=first_omega_0)
        ] + [
            SineLayer(hidden_features,
                      hidden_features,
                      keys[i + 1],
                      omega_0=hidden_omega_0) for i in range(hidden_layers)
        ] + [Linear(hidden_features, out_features, keys[-1], False)]

    def single_call(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        return x


class MLPComposer(MLP):
    mlps: list[MLP]

    def __init__(self, key: jax.random.PRNGKey, mlp_types, mlp_cfgs):
        keys = jax.random.split(key, len(mlp_types))

        self.mlps = [
            globals()[mlp_type](**mlp_cfg, key=subkey)
            for (mlp_type, mlp_cfg, subkey) in zip(mlp_types, mlp_cfgs, keys)
        ]

    def single_call(self, x):
        return jnp.concatenate([mlp.single_call(x) for mlp in self.mlps])

    def get_aux_loss(self):
        return jnp.array([mlp.get_aux_loss() for mlp in self.mlps]).sum()
