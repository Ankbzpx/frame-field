import igl
from common import tet_from_grid
from jax.experimental import sparse

import jax
from jax import vmap, numpy as jnp, jacfwd, jit, hessian
import equinox as eqx
from jaxtyping import Array

from icecream import ic


class MLP(eqx.Module):

    def __init__():
        pass

    def single_call(self, x, z):
        pass

    def single_call_split(self, x, z):
        x = self.single_call(x, z)
        return x[0], x[1:]

    def single_call_aux(self, x, z):
        x = self.single_call(x, z)
        return x[1:]

    def single_call_grad(self, x, z):
        return eqx.filter_value_and_grad(self.single_call_split,
                                         has_aux=True)(x, z)

    def single_call_jac(self, x, z):

        def __single_call(x, z):
            val = self.single_call(x, z)
            return val, val

        return jacfwd(__single_call, has_aux=True)(x, z)

    def single_call_lap(self, x, z):

        def __single_call(x, z):
            return self.single_call(x, z)[0]

        return jnp.trace(hessian(__single_call)(x, z))

    def call_aux(self, x, z):
        return vmap(self.single_call_aux)(x, z)

    def call_grad(self, x, z):
        return vmap(self.single_call_grad)(x, z)

    def call_grad_param(self, x, z, param_func):
        (sdf, aux), normal = vmap(self.single_call_grad)(x, z)
        aux_param = vmap(param_func)(aux)
        return (sdf, aux_param), normal

    def call_jac(self, x, z):
        return vmap(self.single_call_jac)(x, z)

    def call_jac_param(self, x, z, param_func):

        def __single_call(x, z):
            (sdf, aux), normal = self.single_call_grad(x, z)
            aux_param = param_func(aux)
            return aux_param, ((sdf, aux), normal)

        return vmap(jacfwd(__single_call, has_aux=True))(x, z)

    def call_lap(self, x, z):
        return vmap(self.single_call_lap)(x, z)

    def __call__(self, x, z):
        x = vmap(self.single_call)(x, z)
        return x

    def get_aux_loss(self):
        return 0


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


class StandardMLP(MLP):
    layers: list[eqx.Module]
    activation: str
    input_scale: float

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu',
                 input_scale: float = 1,
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation
        self.input_scale = input_scale

        self.layers = [
            Linear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            Linear(hidden_features, hidden_features, keys[i + 1], xavier_init)
            for i in range(hidden_layers)
        ] + [Linear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x, z):
        x = jnp.hstack([self.input_scale * x, z])
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)
        return x


class ResMLP(MLP):
    layers: list[eqx.Module]
    activation: str
    input_scale: float

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='elu',
                 input_scale: float = 1,
                 **kwargs):
        keys = jax.random.split(key, 2 * hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation
        self.input_scale = input_scale

        self.layers = [
            Linear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            Linear(hidden_features, hidden_features, keys[i + 1], xavier_init)
            for i in range(2 * hidden_layers)
        ] + [Linear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x, z):
        activation = getattr(jax.nn, self.activation)

        x = jnp.hstack([self.input_scale * x, z])
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
    input_scale: float

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 activation='tanh',
                 input_scale: float = 1,
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == 'tanh'
        self.activation = activation
        self.input_scale = input_scale

        self.layers = [
            LipLinear(in_features, hidden_features, keys[0], xavier_init)
        ] + [
            LipLinear(hidden_features, hidden_features, keys[i + 1],
                      xavier_init) for i in range(hidden_layers)
        ] + [LipLinear(hidden_features, out_features, keys[-1], xavier_init)]

    def single_call(self, x, z):
        x = jnp.hstack([self.input_scale * x, z])
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
    input_scale: float

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 hidden_layers: int,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 first_omega_0: float = 30,
                 hidden_omega_0: float = 30,
                 input_scale: float = 1,
                 **kwargs):
        keys = jax.random.split(key, hidden_layers + 2)
        self.input_scale = input_scale

        # Section 3.2
        # For [-1, 1], first_omega_0 span it to [-30, 30]
        # Here to scale periods back
        first_omega_0 = first_omega_0 / input_scale

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

    def single_call(self, x, z):
        x = jnp.hstack([self.input_scale * x, z])
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

    def single_call(self, x, z):
        return jnp.concatenate([mlp.single_call(x, z) for mlp in self.mlps])

    def single_call_aux(self, x, z):
        return jnp.concatenate([mlp.single_call(x, z) for mlp in self.mlps[1:]])

    def get_aux_loss(self):
        return jnp.array([mlp.get_aux_loss() for mlp in self.mlps]).sum()

    def single_call_lap(self, x, z):

        def __single_call(x, z):
            return self.mlps[0].single_call(x, z)[0]

        return jnp.trace(hessian(__single_call)(x, z))


@jit
def curl(jac):
    return jnp.array(
        [jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])


class MLPComposerCurl(MLPComposer):

    def __init__(self, key: jax.random.PRNGKey, mlp_types, mlp_cfgs):
        super().__init__(key, mlp_types, mlp_cfgs)

    # For simplicity, assume the first is sdf, the second is vector potential
    def _single_call_grad(self, x, z):
        (sdf, _), normal = self.mlps[0].single_call_grad(x, z)
        jac, vec_potential = self.mlps[1].single_call_jac(x, z)
        tangent = curl(jac)
        aux = jnp.hstack([normal, tangent] +
                         [mlp.single_call(x, z) for mlp in self.mlps[2:]])
        return (sdf, aux), normal, vec_potential

    def single_call_grad(self, x, z):
        return self._single_call_grad(x, z)[:-1]

    def single_call(self, x, z):
        (sdf, aux), _ = self.single_call_grad(x, z)
        return jnp.hstack([sdf, aux])

    def call_jac_param(self, x, z, param_func):

        def __single_call(x, z):
            (sdf, aux), normal, vec_potential = self._single_call_grad(x, z)
            potential = jax.lax.stop_gradient(jnp.linalg.norm(vec_potential))
            aux_param = param_func(aux)
            return aux_param, ((sdf, aux), normal)

        return vmap(jacfwd(__single_call, has_aux=True))(x, z)


class RegularGrid(MLP):
    input_scale: float
    res: float
    grid_val: Array
    L: Array

    def __init__(self,
                 out_features: int,
                 key: jax.random.PRNGKey,
                 res=100,
                 input_scale: float = 1,
                 **kwargs):
        self.input_scale = input_scale
        self.res = res

        # TODO: Support anisotropic grid scale
        self.grid_val = jax.random.normal(key, (res, res, res, out_features))

        # FIXME: Avoid putting heavy computation in constructor because it will be called multiple times in equinox
        V, T = tet_from_grid(res)
        L = igl.cotmatrix(V, T)
        self.L = sparse.BCOO.from_scipy_sparse(-L)

    def single_call(self, x, z):
        axis = jnp.linspace(-1.0, 1.0, self.res)
        interp = jax.scipy.interpolate.RegularGridInterpolator(
            (axis, axis, axis), self.grid_val)
        return interp(self.input_scale * jnp.array([x[1], x[0], x[2]]))[0]

    def get_aux_loss(self):
        L = jax.lax.stop_gradient(self.L)
        vertex_attri = self.grid_val.reshape(len(L), -1)
        loss_smooth = jnp.trace(vertex_attri.T @ L @ vertex_attri)
        return loss_smooth
