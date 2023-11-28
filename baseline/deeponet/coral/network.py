
from functools import partial
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import math
from torch.nn import init
from collections import OrderedDict

SCALE = 30.0  # 120.0


class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class Sine(nn.Module):
    def __init__(self, scale=30.0):
        super().__init__()
        self.scale = scale

    def forward(self, input, sine_cos=False):  # [4096, 20480, 1]  -> 20480 = 256 * 8 * 10
        if not sine_cos:
            return torch.sin(input * self.scale)
        else:
            x = torch.stack((torch.sin(input * self.scale), torch.cos(input * self.scale)), dim=1)
            return x.reshape(x.size(0), -1, 1)


nls = {'relu': partial(nn.ReLU),
       'sigmoid': partial(nn.Sigmoid),
       'tanh': partial(nn.Tanh),
       'selu': partial(nn.SELU),
       'softplus': partial(nn.Softplus),
       'gelu': partial(nn.GELU),
       'swish': partial(Swish),
       'sine': partial(Sine, scale=SCALE),
       'elu': partial(nn.ELU)}


class GeneralizedBilinear(nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, mode: str, device=None, dtype=None) -> None:
        """
        x2T W x1 + x2T A + B x1 + bias
        x2: code, x1: spatial coordinates
        mode: a string of length 4, each element is either '0' or '1', '0' means the corresponding term is not used, '1' means used
              For example, '1111' means all terms are enabled. '0011' means the hypernet is disabled, equivalent to a linear layer.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(GeneralizedBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        assert any(c in ['0', '1'] for c in mode) and len(mode) == 4, 'mode should be 4 characters'
        self.mode = mode
        if mode[0] == '1':
            self.W = Parameter(torch.empty(out_features, in1_features, in2_features, **factory_kwargs))
        else:
            self.register_parameter('W', None)
        
        if mode[1] == '1':
            self.A = Parameter(torch.empty(out_features, in2_features, **factory_kwargs))
        else:
            self.register_parameter('A', None)

        if mode[2] == '1':
            self.B = Parameter(torch.empty(out_features, in1_features, **factory_kwargs))
        else:
            self.register_parameter('B', None)

        if mode[3] == '1':
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        if self.mode[0] == '1':
            init.uniform_(self.W, -bound, bound)
        if self.mode[1] == '1':
            init.uniform_(self.A, -bound, bound)
        if self.mode[2] == '1':
            init.uniform_(self.B, -bound, bound)
        if self.mode[3] == '1':
            init.uniform_(self.bias, -bound, bound)

    def get_hypernet_params(self) -> Tensor:
        hypernet_params = [p for p in [self.W, self.A] if p is not None]
        hypernet_params = [p.reshape(-1, self.in2_features) for p in hypernet_params]
        return torch.cat(hypernet_params, dim=0)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: b, t, h, w, s, i
        # input2: b, t, s, j
        # W: o, i, j
        # B: o, i
        # A: o, j
        # bias: o
        res = 0.
        if self.mode[0] == '1':
            weight_code = torch.einsum('btsj,oij->btsoi', input2, self.W)
            linear_trans_1 = torch.einsum('bthwsi,btsoi->bthwso', input1, weight_code)
            res += linear_trans_1
        
        if self.mode[2] == '1':
            linear_trans_2 = torch.einsum('bthwsi,oi->bthwso', input1, self.B)
            res += linear_trans_2
        
        if self.mode[1] == '1':
            bias_code = torch.einsum('btsj,oj->btso', input2, self.A)
            bias_code = bias_code.unsqueeze(2).unsqueeze(2)
            res += bias_code

        if self.mode[3] == '1':
            res += self.bias
        return res
        
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None)


class CodeBilinear(nn.Module):
    __constants__ = ['in1_features', 'in2_features', 'out_features']
    in1_features: int
    in2_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        """
        x2T A + B x1
        x2: code, x1: spatial coordinates
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CodeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.A = Parameter(torch.empty(out_features, in2_features, **factory_kwargs))
        self.B = Parameter(torch.empty(out_features, in1_features, **factory_kwargs))
        self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in1_features)
        init.kaiming_uniform_(self.A, a=math.sqrt(5))
        init.kaiming_uniform_(self.B, a=math.sqrt(5))
        init.uniform_(self.bias, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        # input1: b, t, h, w, s, i
        # input2: b, t, s, j
        # W: o, i, j
        # B: o, i
        # A: o, j
        # bias: o
        res = 0
        
        bias_code = torch.einsum('btsj,oj->btso', input2, self.A)
        bias_code = bias_code.unsqueeze(2).unsqueeze(2)

        linear_trans_2 = torch.einsum('bthwsi,oi->bthwso', input1, self.B)
       
        res += linear_trans_2 
        res += bias_code
        res += self.bias
        return res
        
    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, out_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.out_features, self.bias is not None)


class MFNBase(nn.Module):
    """
    Multiplicative filter network base class.
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers, weight_scale, bias=True, output_act=False, freq_mod=False):
        super().__init__()
        self.first = 3
        self.bilinear = nn.ModuleList(
            [CodeBilinear(in_size, code_size, hidden_size)] +
            [CodeBilinear(hidden_size, code_size, hidden_size) for _ in range(int(n_layers))]
        )
        self.output_bilinear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act
        self.freq_mod = freq_mod
        return

    def forward(self, x, code):
        out = self.filters[0](x) * self.bilinear[0](x*0., code) if not self.freq_mod else self.filters[0](x, code) * self.bilinear[0](x*0., code)
        for i in range(1, len(self.filters)):
            out = self.filters[i](x) * self.bilinear[i](out, code) if not self.freq_mod else self.filters[i](x, code) * self.bilinear[i](out, code)
        out = self.output_bilinear(out)

        if self.output_act:
            out = torch.sin(out)

        if out.shape[-1] == 1:
            out = out.squeeze(-1)

        return out, x


class FourierLayer(nn.Module):
    """
    Sine filter as used in FourierNet.
    """
    def __init__(self, in_features, out_features, weight_scale, bias=False):
        super().__init__()
        self.register_buffer('weight', torch.empty((out_features, in_features)))
        
        self.bias = None
        if bias == True:
            self.bias = Parameter(torch.empty((out_features)))
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight.data *= self.weight_scale
        if self.bias is not None:
            self.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        if self.bias is None:
            return torch.cat([torch.sin(F.linear(x, self.weight)), torch.cos(F.linear(x, self.weight))], dim=-1) # * self.weight_scale
        else:
            return torch.sin(F.linear(x, self.weight, self.bias))


class FourierLayerCode(nn.Module):
    """
    Sine filter as used in FourierNet.
    """
    def __init__(self, in1_features, in2_features, out_features, weight_scale, bias=False):
        super().__init__()
        self.weight = GeneralizedBilinear(in1_features, in2_features, out_features, mode='1010')
        self.weight_scale = weight_scale
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight.B, a=math.sqrt(5))
        self.weight.B.data *= self.weight_scale

    def forward(self, x, code):
        return torch.cat([torch.sin(self.weight(x, code)), torch.cos(self.weight(x, code))], dim=-1)
        
        
class FourierNet(MFNBase):
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, weight_scale=1.0,
                 bias=True, output_act=False, freq_mod=False, **kwargs):
        super().__init__(in_size, hidden_size, code_size, out_size, n_layers, weight_scale, bias, output_act, freq_mod)
        if freq_mod:
            self.filters = nn.ModuleList(
                [FourierLayerCode(in_size, code_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for i in range(n_layers + 1)])
        else:
            self.filters = nn.ModuleList(
                [FourierLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1)) for i in range(n_layers + 1)])
    
    def get_filters_weight(self):
        weights = list()
        for ftr in self.filters:
            weights.append(ftr.weight)
        return torch.cat(weights)


class GaborLayer(nn.Module):
    """
    Gabor-like filter as used in GaborNet.
    """
    def __init__(self, in_features, out_features, weight_scale, alpha=1.0, beta=1.0, bias=False):
        super().__init__()
        self.weight = Parameter(torch.empty((out_features, in_features)))
        self.bias = None
        if bias == True:
            self.bias = Parameter(torch.empty((out_features)))
        self.weight_scale = weight_scale# gamma

        self.mu = nn.Parameter(2 * torch.rand(out_features, in_features) - 1)
        self.gamma = nn.Parameter(
            torch.distributions.gamma.Gamma(alpha, beta).sample((out_features,))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias.data.uniform_(-np.pi, np.pi)

    def forward(self, x):
        D = ((x ** 2).sum(-1)[..., None] + (self.mu ** 2).sum(-1)[None, :] - 2 * x @ self.mu.T)
        if self.bias is None:
            return torch.cat([
                    torch.sin(F.linear(x, self.weight * self.weight_scale)) * torch.exp(-0.5 * D * self.gamma[None, :]),
                    torch.cos(F.linear(x, self.weight * self.weight_scale)) * torch.exp(-0.5 * D * self.gamma[None, :])
                ], dim=-1)
        else:
            return torch.sin(F.linear(x, self.weight, self.bias)) * torch.exp(-0.5 * D * self.gamma[None, :])


class GaborNet(MFNBase):
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, input_scale=256.0, weight_scale=1.0,
                 alpha=6.0, beta=1.0, bias=False, output_act=False, **kwargs):
        super().__init__(in_size, hidden_size, code_size, out_size, n_layers, weight_scale, bias, output_act)
        self.filters = nn.ModuleList([
            GaborLayer(in_size, hidden_size // 2, input_scale / np.sqrt(n_layers + 1), alpha / (n_layers + 1), beta) for _ in range(n_layers + 1)])


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-2)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / SCALE, np.sqrt(6 / num_input) / SCALE)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-2)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


class SineLayer(nn.Module):
    """
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    hyperparameter.
    If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    
class Siren(nn.Module):
    def __init__(self, in_size, hidden_size, code_size, out_size, n_layers=3, first_omega_0=30, hidden_omega_0=30., outermost_linear=True):
        super().__init__()
        self.net = []
        self.net.append(SineLayer(in_size + code_size, hidden_size, is_first=True, omega_0=first_omega_0))
        for i in range(n_layers):
            self.net.append(SineLayer(hidden_size, hidden_size, is_first=False, omega_0=hidden_omega_0))
        if outermost_linear:
            final_linear = nn.Linear(hidden_size, out_size)
            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_size) / hidden_omega_0, np.sqrt(6 / hidden_size) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_size, out_size, is_first=False, omega_0=hidden_omega_0))
        self.net = nn.ModuleList(self.net)
    
    def forward(self, x):
        for layer in self.net[:-1]:
            x = layer(x)
        output = self.net[-1](x)
        return (output, None)


class MLP(nn.Module):
    def __init__(self, code_size, hidden_size, out_size=None, nl='swish'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size),
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, code_size if out_size == None else out_size),
        )

    def forward(self, x):
        return self.net(x)



class MLP_alt(nn.Module):
    def __init__(self, code_size, hidden_size, nl='swish'):
        super().__init__()

        self.net_a = nn.Sequential(
            nn.Linear(code_size // 2, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size),
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, code_size // 2),
        )

        self.net_b = nn.Sequential(
            nn.Linear(code_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size),
            nls[nl](), 
            nn.Linear(hidden_size, hidden_size), 
            nls[nl](), 
            nn.Linear(hidden_size, code_size // 2),
        )

    def forward(self, x):
        dim = x.size(-1)
        x_a = x[..., :dim // 2]
        return torch.cat((self.net_a(x_a), self.net_b(x)), dim=-1)

class SetEncoder(nn.Module):
    def __init__(self, code_size, n_cond, hidden_size, out_size=None, nl='swish'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(code_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, hidden_size),
            nls[nl](),
            nn.Linear(hidden_size, code_size if out_size == None else out_size),
        )
        self.ave = nn.Conv1d(code_size, code_size, n_cond)

    def forward(self, x):
        aggreg = self.net(x)
        return self.ave(aggreg.permute(0, 2, 1)).permute(0, 2, 1).squeeze()

class ConvNet(nn.Module):
    def __init__(self, in_features, hidden_features, nl='relu', padding='circular', kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=kernel_size, padding="same", padding_mode=padding),
            nls[nl](), 
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, padding="same", padding_mode=padding),
            nls[nl](), 
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, padding="same", padding_mode=padding),
            nls[nl](), 
            nn.Conv2d(hidden_features, in_features, kernel_size=kernel_size, padding="same", padding_mode=padding),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.net(x).permute(0, 2, 3, 1)


class MFNBaseOriginal(nn.Module):
    """
    Multiplicative filter network base class.
    Expects the child class to define the 'filters' attribute, which should be 
    a nn.ModuleList of n_layers+1 filters with output equal to hidden_size.
    """

    def __init__(
        self, hidden_size, out_size, n_layers, weight_scale, bias=True, output_act=False
    ):
        super().__init__()

        self.linear = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size, bias) for _ in range(n_layers)]
        )
        self.output_linear = nn.Linear(hidden_size, out_size)
        self.output_act = output_act

        for lin in self.linear:
            lin.weight.data.uniform_(
                -np.sqrt(weight_scale / hidden_size),
                np.sqrt(weight_scale / hidden_size),
            )

        return

    def forward(self, x):
        coords = x[..., :3]
        out = self.filters[0](x)
        for i in range(1, len(self.filters)):
            out = self.filters[i](coords) * self.linear[i - 1](out)
        out = self.output_linear(out)

        if self.output_act:
            out = torch.sin(out)

        return out, x


class FourierLayerOriginal(nn.Module):
    """
    Sine filter as used in FourierNet.
    """

    def __init__(self, in_features, out_features, code_size, weight_scale, layer):
        super().__init__()
        self.linear = nn.Linear(in_features + code_size if layer == 0 else in_features, out_features)
        self.linear.weight.data *= weight_scale  # gamma
        self.linear.bias.data.uniform_(-np.pi, np.pi)
        return

    def forward(self, x):
        return torch.sin(self.linear(x))


class FourierNetOriginal(MFNBaseOriginal):
    def __init__(
        self,
        in_size,
        hidden_size,
        code_size,
        out_size,
        n_layers=3,
        input_scale=256.0,
        weight_scale=1.0,
        bias=True,
        output_act=False,
    ):
        super().__init__(
            hidden_size, out_size, n_layers, weight_scale, bias, output_act
        )
        self.filters = nn.ModuleList(
            [
                FourierLayerOriginal(in_size, hidden_size, code_size, input_scale / np.sqrt(n_layers + 1), layer)
                for layer in range(n_layers + 1)
            ]
        )


class UNet(nn.Module):

    def __init__(self, state_channels=3, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(state_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=state_channels, kernel_size=1
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1).permute(0, 2, 3, 1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )