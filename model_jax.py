import jax
from jax import vmap, numpy as jnp
import equinox as eqx
from jaxtyping import Array

from icecream import ic


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
    layers: list
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu'):
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
        return x[0], x[1:]

    def call_grad(self, x):
        return vmap(eqx.filter_value_and_grad(self.single_call,
                                              has_aux=True))(x)

    def __call__(self, x):
        x = vmap(self.single_call)(x)
        return x


class ResMLP(eqx.Module):
    layers: list
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu'):

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
        return x[0], x[1:]

    def call_grad(self, x):
        return vmap(eqx.filter_value_and_grad(self.single_call,
                                              has_aux=True))(x)

    def __call__(self, x):
        x = vmap(self.single_call)(x)
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


class LipMLP(eqx.Module):
    layers: list
    activation: str

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='tanh'):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation

        self.layers = [
            LipLinear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            LipLinear(hidden_features, hidden_features, keys[i + 1],
                      xavier_init) for i in range(hidden_layers)
        ] + [LipLinear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x, z):
        x = jnp.concatenate([x, z])
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)
        return x[0]

    def __call__(self, x, z):
        x = vmap(self.single_call, in_axes=(0, None))(x, z)
        return x

    def get_lipschitz_loss(self):
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


class Siren(eqx.Module):
    layers: list

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
        return x[0], x[1:]

    def call_grad(self, x):
        return vmap(eqx.filter_value_and_grad(self.single_call,
                                              has_aux=True))(x)

    def __call__(self, x):
        x = vmap(self.single_call)(x)
        return x


if __name__ == '__main__':
    from icecream import ic

    key_model, key_data = jax.random.split(jax.random.PRNGKey(1), 2)
    model = Siren(3, 256, 4, 10, key_model)

    x = jax.random.uniform(key_data, (10, 3))
    (sdf, sh9), normal = model.call_grad(x)

    ic(jnp.abs(sdf).max())
    ic(jnp.abs(sdf).min())
