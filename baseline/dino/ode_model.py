# Copyright 2022 Yuan Yin & Matthieu Kirchmeyer

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from dino.network import MLP, FourierNet
from torch import nn
import einops

from coral.siren import ModulatedSiren

class Derivative(nn.Module):
    def __init__(self, state_c, code_c, hidden_c, **kwargs):
        super().__init__()
        input_dim = code_c * state_c
        self.net = MLP(input_dim, hidden_c, nl="swish")

    def forward(self, t, u):
        return self.net(u)


class Decoder(nn.Module):
    def __init__(self, state_c, hidden_c, code_c, coord_dim, n_layers, model, **kwargs):
        super().__init__()
        self.state_c = state_c
        self.hidden_c = hidden_c
        self.coord_dim = coord_dim
        self.out_dim = 1  # TODO
        self.code_dim = code_c
        self.model_type = model

        if model == 'mfn':
            self.net = FourierNet(
                self.coord_dim,
                self.hidden_c,
                self.code_dim,
                self.out_dim,
                n_layers,
                input_scale=64,
            )
        elif model == 'siren':
            self.net = ModulatedSiren(
                self.coord_dim,
                dim_hidden=self.hidden_c,
                dim_out=self.out_dim,
                num_layers=n_layers,
                w0=10,
                w0_initial=10,
                use_bias=True,
                modulate_scale=False,
                modulate_shift=True,
                use_latent=True,
                latent_dim=self.code_dim,
                modulation_net_dim_hidden=128,
                modulation_net_num_layers=1,
                last_activation=None,
            )
        else:
            NotImplementedError('INR type not implemented')

    def forward(self, x, codes=None):

        if self.model_type == 'mfn':
            if codes is None:
                return self.net(x)
            return self.net(x, codes)
        elif self.model_type == 'siren':
            batch, time, spatial, channel, dim = x.shape

            # x = B, T, X, C, grid
            # codes = B, T, C, L

            x = einops.rearrange(x, 'B T ... C  -> B T (...) C')
            x = einops.rearrange(x, 'B T X C  -> (B T) X C')
            # x = B, X, grid

            codes = einops.rearrange(codes, 'B T C L  -> (B T C) L')

            # x doit etre B, X, grid, codes doit etre B, L
            out = self.net.modulated_forward(x, codes)
            # out est B, X, out

            out = einops.rearrange(out, '(B T) X C -> B T X C', T=time)

            # out doit etre B, T, XY, C
            return out, None
        else:
            NotImplementedError(
                f'model_type {self.model_type} not implemented ')
