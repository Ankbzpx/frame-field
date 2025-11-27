from common import normalize

import equinox as eqx
import igl
import jax
from jax import hessian, jacfwd, numpy as jnp, vmap
from jaxtyping import Array

from icecream import ic


# For abstraction convenience
jax.nn.sin = jnp.sin
jnp.identity = lambda x: x
jnp.normalize = normalize


class MLP(eqx.Module):
    def __init__():
        pass

    def single_call(self, x):
        x = self.input_scale * x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)

        x = getattr(jnp, self.final_activation)(x)
        return x

    def single_call_split(self, x):
        x = self.single_call(x)
        return x[0], x[1:]

    def single_call_aux(self, x):
        x = self.single_call(x)
        return x[1:]

    def single_call_grad(self, x):
        return eqx.filter_value_and_grad(self.single_call_split, has_aux=True)(x)

    def single_call_jac(self, x):
        def __single_call(x):
            val = self.single_call(x)
            return val, val

        return jacfwd(__single_call, has_aux=True)(x)

    def single_call_hessian(self, x):
        def __single_call(x):
            return self.single_call(x)[0]

        return hessian(__single_call)(x)

    def call_aux(self, x):
        return vmap(self.single_call_aux)(x)

    def call_grad(self, x):
        return vmap(self.single_call_grad)(x)

    def call_grad_param(self, x, param_func):
        (sdf, aux), normal = vmap(self.single_call_grad)(x)
        aux_param = vmap(param_func)(aux)
        return (sdf, aux_param), normal

    def call_jac(self, x):
        return vmap(self.single_call_jac)(x)

    def call_jac_param(self, x, param_func):
        def __single_call(x):
            (sdf, aux), normal = self.single_call_grad(x)
            aux_param = param_func(aux)
            return aux_param, ((sdf, aux), normal)

        return vmap(jacfwd(__single_call, has_aux=True))(x)

    # WARNING: This is slower than call 'call_hessian' and 'call_grad' separately
    def call_hessian_aux(self, x):
        def __single_call(x):
            (sdf, aux), normal = self.single_call_grad(x)
            return normal, ((sdf, aux), normal)

        return vmap(jacfwd(__single_call, has_aux=True))(x)

    def call_hessian(self, x):
        return vmap(self.single_call_hessian)(x)

    def call_laplacian(self, x):
        return vmap(jnp.trace)(self.call_hessian(x))

    def __call__(self, x):
        x = vmap(self.single_call)(x)
        return x

    def get_aux_loss(self):
        return 0


class Linear(eqx.Module):
    W: Array
    b: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        xavier_init: bool = False,
    ):
        if xavier_init:
            self.W = jax.random.uniform(
                key, (out_features, in_features), minval=-1.0, maxval=1.0
            ) * jnp.sqrt(6.0 / (in_features + out_features))
        else:
            self.W = jax.random.normal(key, (out_features, in_features)) * jnp.sqrt(
                2.0 / in_features
            )

        self.b = jax.random.uniform(
            key, (out_features,), minval=-1.0, maxval=1.0
        ) * jnp.sqrt(1 / in_features)

    def __call__(self, x):
        return self.W @ x + self.b


class StandardMLP(MLP):
    layers: list[eqx.Module]
    activation: str
    input_scale: float
    final_activation: str

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        key: jax.random.PRNGKey,
        activation="elu",
        input_scale: float = 1,
        final_activation="identity",
        **kwargs,
    ):
        keys = jax.random.split(key, hidden_layers + 2)

        xavier_init = activation == "tanh"
        self.activation = activation
        self.input_scale = input_scale
        self.final_activation = final_activation

        self.layers = (
            [Linear(in_features, hidden_features, keys[0], xavier_init)]
            + [
                Linear(hidden_features, hidden_features, keys[i + 1], xavier_init)
                for i in range(hidden_layers)
            ]
            + [Linear(hidden_features, out_features, keys[-1], xavier_init)]
        )


# This can in theory be merged with `Linear`, but we rely `omega_0` for lr filtering
class SineLayer(eqx.Module):
    W: Array
    b: Array
    omega_0: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        is_first: bool = False,
        is_last: bool = False,
        omega_0: float = 30.0,
    ):
        self.omega_0 = 1.0 if is_last else omega_0

        if is_first:
            self.W = (
                jax.random.uniform(
                    key, (out_features, in_features), minval=-1.0, maxval=1.0
                )
                / in_features
            )
        else:
            self.W = (
                jax.random.uniform(
                    key, (out_features, in_features), minval=-1.0, maxval=1.0
                )
                * jnp.sqrt(6 / in_features)
                / omega_0
            )

        self.b = jax.random.uniform(
            key, (out_features,), minval=-1.0, maxval=1.0
        ) * jnp.sqrt(1 / in_features)

    def __call__(self, x):
        return self.omega_0 * (self.W @ x + self.b)


# Reference: https://github.com/Chumbyte/DiGS/blob/44bcc890b8519e89263411476c892e671a024151/models/DiGS.py#L111
class GeomSineLayer(eqx.Module):
    W: Array
    b: Array
    omega_0: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        is_second_last=False,
        omega_0: float = 30,
    ):
        # Workaround "Results do not match the reference. This is likely a bug/unexpected loss of precision."
        # FIXME This shouldn't be necessary
        self.omega_0 = jnp.asarray(omega_0)

        if is_second_last:
            self.W = 0.5 * jnp.pi * jnp.eye(
                out_features, in_features
            ) / omega_0 + 1e-3 * jax.random.normal(key, (out_features, in_features))
            self.b = 0.5 * jnp.pi * jnp.ones(
                out_features
            ) / omega_0 + 1e-3 * jax.random.normal(key, (out_features,))
        else:
            self.W = (
                jax.random.uniform(
                    key, (out_features, in_features), minval=-1.0, maxval=1.0
                )
                * jnp.sqrt(3 / out_features)
                / omega_0
            )
            # Small Gaussian noise to facilitate learning
            self.b = (
                jnp.zeros(out_features)
                + jax.random.uniform(key, (out_features,), minval=-1.0, maxval=1.0)
                / (out_features * 1000)
                / omega_0
            )

    def __call__(self, x):
        return self.omega_0 * (self.W @ x + self.b)


class MFGILayer(eqx.Module):
    W: Array
    b: Array
    omega_0: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        is_first: bool = False,
        omega_0: float = 30,
        low_freq_portion: float = 0.25,
    ):
        # Workaround " Results do not match the reference. This is likely a bug/unexpected loss of precision."
        # FIXME This shouldn't be necessary
        self.omega_0 = jnp.asarray(omega_0)

        low_freq_features = jnp.int32(out_features * low_freq_portion)
        high_freq_features = out_features - low_freq_features

        if is_first:
            W_low_freq = (
                jax.random.uniform(
                    key, (low_freq_features, in_features), minval=-1.0, maxval=1.0
                )
                * jnp.sqrt(3 / in_features)
                / omega_0
            )
            W_high_freq = jax.random.uniform(
                key, (high_freq_features, in_features), minval=-1.0, maxval=1.0
            ) * jnp.sqrt(3 / in_features)

            self.W = jnp.vstack([W_low_freq, W_high_freq])
        else:
            # TODO 1e-3 in paper, but 5e-4 in code. Why scales down?
            W = (
                jax.random.uniform(
                    key, (out_features, in_features), minval=-1.0, maxval=1.0
                )
                * jnp.sqrt(3 / in_features)
                / omega_0
                * 5e-4
            )

            # FIXME Handle out of bound cases
            self.W = W.at[:low_freq_features, :low_freq_features].set(
                jax.random.uniform(
                    key, (low_freq_features, low_freq_features), minval=-1.0, maxval=1.0
                )
                * jnp.sqrt(3 / in_features)
                / omega_0
            )

        self.b = (
            jnp.zeros(out_features)
            + jax.random.uniform(key, (out_features,), minval=-1.0, maxval=1.0)
            / (out_features * 1000)
            / omega_0
        )

    def __call__(self, x):
        return self.omega_0 * (self.W @ x + self.b)


class GeoSineLast(eqx.Module):
    W: Array
    b: Array

    def __init__(self, in_features: int, out_features: int, key: jax.random.PRNGKey):
        self.W = -jnp.ones((out_features, in_features)) + 1e-5 * jax.random.normal(
            key, (out_features, in_features)
        )
        self.b = jnp.ones(out_features) * in_features

    def __call__(self, x):
        return self.W @ x + self.b


class Siren(MLP):
    layers: list[eqx.Module]
    activation: str
    input_scale: float
    r_sphere: bool
    final_activation: str

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        key: jax.random.PRNGKey,
        first_omega_0: float = 30,
        hidden_omega_0: float = 30,
        input_scale: float = 1,
        init_method="default",
        final_activation="identity",
        **kwargs,
    ):
        keys = jax.random.split(key, hidden_layers + 2)
        self.input_scale = input_scale
        self.activation = "sin"
        self.final_activation = final_activation

        if init_method == "geom":
            self.r_sphere = True
            self.layers = (
                [
                    GeomSineLayer(
                        in_features, hidden_features, keys[0], omega_0=first_omega_0
                    ),
                ]
                + [
                    GeomSineLayer(
                        hidden_features,
                        hidden_features,
                        keys[1 + i],
                        omega_0=hidden_omega_0,
                    )
                    for i in range(hidden_layers - 1)
                ]
                + [
                    GeomSineLayer(
                        hidden_features,
                        hidden_features,
                        keys[-2],
                        omega_0=hidden_omega_0,
                        is_second_last=True,
                    )
                ]
                + [GeoSineLast(hidden_features, out_features, keys[-1])]
            )
        elif init_method == "mfgi":
            self.r_sphere = True
            self.layers = (
                [
                    MFGILayer(
                        in_features,
                        hidden_features,
                        keys[0],
                        is_first=True,
                        omega_0=first_omega_0,
                    )
                ]
                + [
                    MFGILayer(
                        hidden_features,
                        hidden_features,
                        keys[1],
                        is_first=False,
                        omega_0=hidden_omega_0,
                    )
                ]
                + [
                    GeomSineLayer(
                        hidden_features,
                        hidden_features,
                        keys[2 + i],
                        omega_0=hidden_omega_0,
                    )
                    for i in range(hidden_layers - 2)
                ]
                + [
                    GeomSineLayer(
                        hidden_features,
                        hidden_features,
                        keys[-2],
                        omega_0=hidden_omega_0,
                        is_second_last=True,
                    )
                ]
                + [GeoSineLast(hidden_features, out_features, keys[-1])]
            )
        else:
            self.r_sphere = False
            self.layers = (
                [
                    SineLayer(
                        in_features,
                        hidden_features,
                        keys[0],
                        is_first=True,
                        omega_0=first_omega_0,
                    )
                ]
                + [
                    SineLayer(
                        hidden_features,
                        hidden_features,
                        keys[i + 1],
                        omega_0=hidden_omega_0,
                    )
                    for i in range(hidden_layers)
                ]
                + [SineLayer(hidden_features, out_features, keys[-1], is_last=True)]
            )

    def single_call(self, x):
        x = self.input_scale * x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(jax.nn, self.activation)(x)

        if self.r_sphere:
            x = jnp.sign(x) * jnp.sqrt(jnp.abs(x) + 1e-8)
            x = 0.1 * (x - 1.6)

        x = getattr(jnp, self.final_activation)(x)
        return x


# Modified from: https://github.com/ml-for-gp/jaxgptoolbox/blob/main/demos/lipschitz_mlp/model.py
class LipLinear(Linear):
    W: Array
    b: Array
    c: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        xavier_init: bool = False,
    ):
        super().__init__(in_features, out_features, key, xavier_init)
        self.c = jnp.max(jnp.sum(jnp.abs(self.W), axis=1))

    # L-infinity weight normalization
    def weight_normalization(self, W, softplus_c):
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]

    def __call__(self, x):
        return self.weight_normalization(self.W, self.lipschitz()) @ x + self.b

    def lipschitz(self):
        return jax.nn.softplus(self.c)


# Sin activation with omega_0=1 is still 1-Lipschitz activation function
class LipSineLayer(SineLayer):
    c: Array
    c_scale: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.random.PRNGKey,
        is_first: bool = False,
        is_last: bool = False,
        omega_0: float = 1,
    ):
        super().__init__(in_features, out_features, key, is_first, is_last, omega_0)
        self.c = jnp.max(jnp.sum(jnp.abs(self.W), axis=1))

        # For initial layers, omega_0 equivalently scales weight by the same amount
        # For rest of layers, scale based on omega_0
        #   as init weights are shrunk by omega_0 (then pre-multiplied before activation) to amplify gradient magnitude
        self.c_scale = 1.0 * omega_0

    # L-infinity weight normalization
    def weight_normalization(self, W, softplus_c):
        absrowsum = jnp.sum(jnp.abs(W), axis=1)
        scale = jnp.minimum(1.0, softplus_c / absrowsum)
        return W * scale[:, None]

    def __call__(self, x):
        return self.omega_0 * (
            self.weight_normalization(self.W, self.lipschitz()) @ x + self.b
        )

    def lipschitz(self):
        # The gradient update of c should match the weight
        return jax.nn.softplus(self.c_scale * self.c)


class LipMLP(MLP):
    layers: list[eqx.Module]
    activation: str
    input_scale: float
    final_activation: str

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        hidden_layers: int,
        out_features: int,
        key: jax.random.PRNGKey,
        first_omega_0: float = 1.0,
        hidden_omega_0: float = 30.0,
        activation="tanh",
        input_scale: float = 1,
        final_activation="identity",
        **kwargs,
    ):
        keys = jax.random.split(key, hidden_layers + 2)

        self.activation = activation
        self.input_scale = input_scale
        self.final_activation = final_activation

        if activation != "sin":
            xavier_init = activation == "tanh"
            self.layers = (
                [LipLinear(in_features, hidden_features, keys[0], xavier_init)]
                + [
                    LipLinear(
                        hidden_features, hidden_features, keys[i + 1], xavier_init
                    )
                    for i in range(hidden_layers)
                ]
                + [LipLinear(hidden_features, out_features, keys[-1], xavier_init)]
            )
        else:
            # With Siren, the initial lipchitz bound should only be scaled by first_omega_0
            self.layers = (
                [
                    LipSineLayer(
                        in_features,
                        hidden_features,
                        keys[0],
                        is_first=True,
                        omega_0=first_omega_0,
                    )
                ]
                + [
                    LipSineLayer(
                        hidden_features,
                        hidden_features,
                        keys[i + 1],
                        omega_0=hidden_omega_0,
                    )
                    for i in range(hidden_layers)
                ]
                + [
                    LipSineLayer(
                        hidden_features,
                        out_features,
                        keys[-1],
                        omega_0=hidden_omega_0,
                        is_last=True,
                    )
                ]
            )

    # Lipschitz loss
    # Reference: https://github.com/ml-for-gp/jaxgptoolbox/blob/7048aada5db1e6603a3d13fb1bc1ee2c61762985/demos/lipschitz_mlp/model.py#L82
    def get_aux_loss(self):
        loss_lip = 1.0
        for layer in self.layers:
            loss_lip = loss_lip * layer.lipschitz()
        return loss_lip


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

    def single_call_aux(self, x):
        return jnp.concatenate([mlp.single_call(x) for mlp in self.mlps[1:]])

    def get_aux_loss(self):
        return jnp.array([mlp.get_aux_loss() for mlp in self.mlps]).sum()

    def single_call_hessian(self, x):
        def __single_call(x):
            return self.mlps[0].single_call(x)[0]

        return hessian(__single_call)(x)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    lipmlp = LipMLP(
        3, 256, 4, 9, key, activation="sin", first_omega_0=1, hidden_omega_0=30
    )
    for idx in range(len(lipmlp.layers)):
        print(idx, lipmlp.layers[idx].lipschitz())
    print("Total: ", lipmlp.get_aux_loss())
