from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


ACTIVATIONS = {
    "relu": partial(nn.ReLU),
    "sigmoid": partial(nn.Sigmoid),
    "tanh": partial(nn.Tanh),
    "selu": partial(nn.SELU),
    "softplus": partial(nn.Softplus),
    "gelu": partial(nn.GELU),
    "swish": partial(Swish),
    "elu": partial(nn.ELU),
    "leakyrelu": partial(nn.LeakyReLU),
}


class ResBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        drop_rate=0.0,
        activation="swish",
    ):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.activation1 = ACTIVATIONS[activation]()
        self.activation2 = ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        eta = self.linear1(x)
        # eta = self.batch_norm1(eta)
        eta = self.linear2(self.activation1(eta))
        # no more dropout
        # out = self.activation2(x + self.dropout(eta))
        out = x + self.activation2(self.dropout(eta))
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
    ):
        super().__init__()
        net = [ResBlock(input_dim, hidden_dim, dropout, activation)]
        for _ in range(depth - 1):
            net.append(ResBlock(input_dim, hidden_dim, dropout, activation))

        self.net = nn.Sequential(*net)
        self.project_map = nn.Linear(input_dim, output_dim)

    def forward(self, z):
        out = self.net(z)
        out = self.project_map(out)

        return out


class ResNetDynamics(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
        dt= 0.5/250
    ):
        super().__init__()
        net = [ResBlock(input_dim, hidden_dim, dropout, activation)]
        for _ in range(depth - 1):
            net.append(ResBlock(input_dim, hidden_dim, dropout, activation))

        self.net = nn.Sequential(*net)
        self.project_map = nn.Linear(input_dim, output_dim)
        self.dt = dt

    def forward(self, z):
        # input (b, l, t)
        T = z.shape[-1]
        z_last = z[..., -1].unsqueeze(-1)
        z = einops.rearrange(z, 'b l t -> b (l t)')
        out = self.net(z)
        out = self.project_map(out)

        out = einops.rearrange(out, 'b (l t) -> b l t', t=T)

        #dt = (torch.ones(1, 1, T) * self.dt).to(out.device)
        #dt = torch.cumsum(dt, dim=2)

        out = z_last + out
        return out  


class MLP(nn.Module):
    def __init__(
        self,
        input_dim=64,
        hidden_dim=64,
        output_dim=64,
        depth=2,
        dropout=0.0,
        activation="swish",
    ):
        super().__init__()
        self.activation = ACTIVATIONS[activation]
        net = [nn.Linear(input_dim, hidden_dim)]
        for _ in range(depth - 1):
            net.append(self.activation())
            net.append(nn.Dropout(dropout))
            net.append(nn.Linear(hidden_dim, hidden_dim))
        net.append(self.activation())

        self.net = nn.Sequential(*net)
        self.dropout = nn.Dropout(dropout)
        self.project_map = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = self.net(z)
        # z = self.dropout(z)
        out = self.project_map(z)

        return out


class MLP2(nn.Module):
    def __init__(self, code_size, hidden_size, depth=1, nl="swish"):
        super().__init__()

        net = [nn.Linear(code_size, hidden_size), ACTIVATIONS[nl]()]

        for j in range(depth - 1):
            net.append(nn.Linear(hidden_size, hidden_size))
            net.append(ACTIVATIONS[nl]())

        net.append(nn.Linear(hidden_size, code_size))
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, depth=2, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP2(input_dim, hidden_c, depth=depth, nl="swish")

    def forward(self, t, u):
        return self.net(u)
