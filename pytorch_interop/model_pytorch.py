import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import tinycudann as tcnn


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i],
                                   x,
                                   torch.ones_like(y[..., i]),
                                   create_graph=True)[0][..., i:i + 1]
    return div


def hessian(y, x):
    grad = gradient(y, x)
    return vector_gradient(grad, x)


def vector_gradient(grad, x):
    return torch.stack([gradient(grad[:, i], x) for i in range(grad.shape[-1])],
                       -1)


def gradient(y, x, grad_outputs=None):

    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


class VanillaMLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 input_scale=1.,
                 **kwargs):
        super().__init__()

        self.input_scale = input_scale
        self.layers = nn.ModuleList(
            [nn.Linear(in_features, hidden_features)] +
            [nn.Linear(hidden_features, hidden_features)] * hidden_layers +
            [nn.Linear(hidden_features, out_features)])

    def forward(self, x):
        x = self.input_scale * x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x


# https://github.com/vsitzmann/siren
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 is_first=False,
                 omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -math.sqrt(6 / self.in_features) / self.omega_0,
                    math.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class Siren(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 input_scale=1.,
                 outermost_linear=True,
                 first_omega_0=30.,
                 hidden_omega_0=30.,
                 **kwargs):
        super().__init__()

        self.input_scale = input_scale
        self.net = []
        self.net.append(
            SineLayer(in_features,
                      hidden_features,
                      is_first=True,
                      omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(hidden_features,
                          hidden_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -math.sqrt(6 / hidden_features) / hidden_omega_0,
                    math.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features,
                          out_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(self.input_scale * coords)
        return output


# https://github.com/HTDerekLiu/LipschitzMLP_SIGGRAPH_Demo/blob/main/model_lipmlp.py
class LipschitzLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        # The reference implementation uses zero bias initialization, which IMO is risky
        # Here we use pytorch Linear with default initialization (i.e. kaiming_uniform)
        self.linear = nn.Linear(in_features, out_features, bias=True)

        c_init = torch.max(torch.sum(torch.abs(self.linear.weight), dim=1))
        self.c = nn.Parameter(c_init)

    def normalization(self, W: torch.Tensor, softplus_c) -> torch.Tensor:
        absrowsum = W.abs().sum(dim=1)
        scale = torch.minimum(torch.ones_like(absrowsum),
                              softplus_c / absrowsum)
        return W * scale[:, None]

    # https://pytorch.org/docs/master/generated/torch.nn.functional.linear.html?highlight=linear#torch.nn.functional.linear
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum(
            'bc,mc->bm', input,
            self.normalization(self.linear.weight, F.softplus(
                self.c))) + self.linear.bias[None, :]


class LipschitzMLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 input_scale=1.,
                 **kwargs):
        super().__init__()

        self.input_scale = input_scale
        self.layers = nn.ModuleList(
            [LipschitzLinear(in_features, hidden_features)] + [
                LipschitzLinear(hidden_features, hidden_features)
                for _ in range(hidden_layers)
            ] + [LipschitzLinear(hidden_features, out_features)])

    def forward(self, x):
        x = self.input_scale * x
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

    def get_lipschitz_loss(self):
        loss_lip = 1.0
        for layer in self.layers:
            loss_lip = loss_lip * F.softplus(layer.c)
        return loss_lip


class HashMLP(nn.Module):

    def __init__(
            self,
            in_features,
            hidden_features,
            hidden_layers,
            out_features,
            input_scale=1.,
            interpolation="Nearest",    #"Linear"
            **kwargs):
        super().__init__()

        self.input_scale = input_scale

        hash_cfg = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 15,
            "base_resolution": 16,
            "per_level_scale": 1.5,
            "interpolation": interpolation
        }
        self.encoding = tcnn.Encoding(in_features, hash_cfg)

        in_dim = hash_cfg["n_levels"] * hash_cfg["n_features_per_level"]
        self.layers = nn.ModuleList([nn.Linear(in_dim, hidden_features)] + [
            nn.Linear(hidden_features, hidden_features)
            for _ in range(hidden_layers)
        ] + [nn.Linear(hidden_features, out_features)])

    def forward(self, x):
        x = self.input_scale * x
        x = self.encoding(x).float()
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        return x

    def val_and_grad(self, x, eps, func_param=lambda x: x):
        eps_x = torch.tensor([eps, 0., 0.], dtype=x.dtype, device=x.device)
        eps_y = torch.tensor([0., eps, 0.], dtype=x.dtype, device=x.device)
        eps_z = torch.tensor([0., 0., eps], dtype=x.dtype, device=x.device)

        # Forward difference
        val = self.octa_mlp(x)
        param = func_param(val)
        param_x = func_param(self.octa_mlp(x + eps_x[None, :]))
        param_y = func_param(self.octa_mlp(x + eps_y[None, :]))
        param_z = func_param(self.octa_mlp(x + eps_z[None, :]))

        dx = (param_x - param) / eps
        dy = (param_y - param) / eps
        dz = (param_z - param) / eps
        grad = torch.stack([dx, dy, dz], dim=-1)

        return val, grad


if __name__ == '__main__':
    import os
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    from icecream import ic

    x = torch.rand(100, 3).cuda()
    x.requires_grad_(True)
    mlp1 = HashMLP(3, 256, 1, 1).cuda()
    mlp2 = HashMLP(3, 256, 1, 3).cuda()

    y = mlp1(x)
    z = mlp2(x)

    ic(gradient(y, x).shape)
    ic(z.shape)
