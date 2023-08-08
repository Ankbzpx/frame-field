import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, grad_and_value
import math
import numpy as np
from collections import OrderedDict
from icecream import ic


# https://colab.research.google.com/github/vsitzmann/siren/blob/master/explore_siren.ipynb
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


def thin_plate(y, x):
    grad = gradient(y, x)
    grad_2 = torch.hstack(
        [gradient(grad_split, x) for grad_split in grad.split(1, -1)])
    return grad_2.square().sum(dim=1)


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x],
                               grad_outputs=grad_outputs,
                               create_graph=True)[0]
    return grad


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
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 outermost_linear=True,
                 first_omega_0=30,
                 hidden_omega_0=30.):
        super().__init__()

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
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(hidden_features,
                          out_features,
                          is_first=False,
                          omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(
            True)    # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join(
                    (str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join(
                (str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


class MLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 activation='elu'):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(in_features, hidden_features)] + [
                nn.Linear(hidden_features, hidden_features)
                for _ in range(hidden_layers)
            ] + [nn.Linear(hidden_features, out_features)])
        self.activation = activation

    def forward_pass(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(F, self.activation)(x)
        return x[0], x[1:]

    def forward(self, coords):
        return vmap(self.forward_pass)(coords)

    def forward_grad(self, coords):
        return vmap(grad_and_value(self.forward_pass, has_aux=True))(coords)

    def forward_grad_batch(self, coords):
        return vmap(self.forward_grad)(coords)


# https://nv-tlabs.github.io/lip-mlp/lipmlp_final.pdf
class LipschitzLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        weight_init = torch.randn(out_features, in_features) * math.sqrt(
            2. / in_features)
        bias_init = torch.zeros(out_features)
        c_init = torch.max(torch.sum(torch.abs(weight_init), dim=1))

        self.weight = nn.Parameter(weight_init)
        self.bias = nn.Parameter(bias_init)
        self.c = nn.Parameter(c_init)

    def normalization(self, W: torch.Tensor,
                      softplus_c: torch.float) -> torch.Tensor:
        absrowsum = W.abs().sum(dim=1)
        scale = torch.minimum(torch.ones_like(absrowsum),
                              softplus_c / absrowsum)
        return W * scale[:, None]

    # https://pytorch.org/docs/master/generated/torch.nn.functional.linear.html?highlight=linear#torch.nn.functional.linear
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bc,mc->bm', input,
                            self.normalization(self.weight, F.softplus(
                                self.c))) + self.bias[None, :]


class LipschitzMLP(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features,
                 hidden_layers,
                 out_features,
                 activation='tanh'):
        super().__init__()

        self.layers = nn.ModuleList(
            [LipschitzLinear(in_features, hidden_features)] + [
                LipschitzLinear(hidden_features, hidden_features)
                for _ in range(hidden_layers)
            ] + [LipschitzLinear(hidden_features, out_features)])
        self.activation = activation

    def forward_pass(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = getattr(F, self.activation)(x)
        return x

    def forward(self, coords):
        coords = coords.clone().detach()
        output = self.forward_pass(coords)
        return output, coords

    def get_lipschitz_loss(self):
        loss_lip = 1.0
        for layer in self.layers:
            loss_lip = loss_lip * F.softplus(layer.c)
        return loss_lip


class RBF(nn.Module):

    def __init__(self, in_features, hidden_features, hidden_layers,
                 out_features):
        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(in_features, hidden_features)] + [
                nn.Linear(hidden_features, hidden_features)
                for _ in range(hidden_layers)
            ] + [nn.Linear(hidden_features, in_features)])

        self.rbf = nn.Parameter(torch.randn((hidden_features, in_features)))
        self.last_layer = nn.Linear(in_features + hidden_features, out_features)

    def forward_pass(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers) - 1:
                x = F.relu(x)

        x_abs = torch.cdist(x, self.rbf)
        phi = x_abs**2 * torch.log(F.softplus(x_abs))

        x = self.last_layer(torch.concat([phi, x], -1))
        return x

    def forward(self, coords):
        coords = coords.clone().detach()
        output = self.forward_pass(coords)
        return output, coords


if __name__ == '__main__':

    mlp = torch.compile(MLP(3, 64, 4, 1), mode="reduce-overhead")
    x = torch.randn(10, 3)
    ic(mlp.forward_grad(x))
