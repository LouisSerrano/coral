from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from coral.siren import LatentToModulation
from coral.utils.film_conditioning import film, film_linear, film_translate

class GaussianEncoding(nn.Module):
    def __init__(self, embedding_size, scale, dims=2, gaussian=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.scale = scale

        if gaussian:
            bvals = torch.randn(embedding_size // 2, dims) * scale
        else:
            bvals = 2.0 ** torch.linspace(0, scale, embedding_size // 2) - 1

            if dims == 1:
                bvals = bvals[:, None]

            elif dims == 2:
                bvals = torch.stack([bvals, torch.zeros_like(bvals)], dim=-1)
                bvals = torch.cat([bvals, torch.roll(bvals, 1, -1)], dim=0)

            else:
                tmp = (dims - 1) * (torch.zeros_like(bvals),)
                bvals = torch.stack([bvals, *tmp], dim=-1)

                tmp = [torch.roll(bvals, i, -1) for i in range(1, dims)]
                bvals = torch.cat([bvals, *tmp], dim=0)

        avals = torch.ones((bvals.shape[0]))
        self.avals = nn.Parameter(avals, requires_grad=False)
        self.bvals = nn.Parameter(bvals, requires_grad=False)

    def forward(self, tensor) -> torch.Tensor:
        """
        Apply positional encoding to the input.
        """

        return torch.cat(
            [
                self.avals * torch.sin((2.0 * np.pi * tensor) @ self.bvals.T),
                self.avals * torch.cos((2.0 * np.pi * tensor) @ self.bvals.T),
            ],
            dim=-1,
        )


class NeRFEncoding(nn.Module):
    """PyTorch implementation of regular positional embedding, as used in the original NeRF and Transformer papers."""

    def __init__(
        self,
        num_freq,
        max_freq_log2,
        log_sampling=True,
        include_input=True,
        input_dim=3,
        base_freq=2,
    ):
        """Initialize the module.
        Args:
            num_freq (int): The number of frequency bands to sample.
            max_freq_log2 (int): The maximum frequency.
                                 The bands will be sampled at regular intervals in [0, 2^max_freq_log2].
            log_sampling (bool): If true, will sample frequency bands in log space.
            include_input (bool): If true, will concatenate the input.
            input_dim (int): The dimension of the input coordinate space.
        Returns:
            (void): Initializes the encoding.
        """
        super().__init__()

        self.num_freq = num_freq
        self.max_freq_log2 = max_freq_log2
        self.log_sampling = log_sampling
        self.include_input = include_input
        self.out_dim = 0
        self.base_freq = base_freq

        if include_input:
            self.out_dim += input_dim

        if self.log_sampling:
            self.bands = self.base_freq ** torch.linspace(
                0.0, max_freq_log2, steps=num_freq
            )
        else:
            self.bands = torch.linspace(
                1, self.base_freq**max_freq_log2, steps=num_freq
            )

        # The out_dim is really just input_dim + num_freq * input_dim * 2 (for sin and cos)
        self.out_dim += self.bands.shape[0] * input_dim * 2
        self.bands = nn.Parameter(self.bands).requires_grad_(False)

    def forward(self, coords, with_batch=True):
        """Embeds the coordinates.
        Args:
            coords (torch.FloatTensor): Coordinates of shape [N, input_dim]
        Returns:
            (torch.FloatTensor): Embeddings of shape [N, input_dim + out_dim] or [N, out_dim].
        """
        if with_batch:
            N = coords.shape[0]
            winded = (coords[..., None, :] * self.bands[None,None,:,None]).reshape(
                N, coords.shape[1], coords.shape[-1] * self.num_freq)
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)

        else:
            N = coords.shape[0]
            winded = (coords[:, None] * self.bands[None, :, None]).reshape(
                N, coords.shape[1] * self.num_freq
            )
            encoded = torch.cat([torch.sin(winded), torch.cos(winded)], dim=-1)
            if self.include_input:
                encoded = torch.cat([coords, encoded], dim=-1)
        return encoded

    def name(self) -> str:
        """A human readable name for the given wisp module."""
        return "Positional Encoding"

    def public_properties(self) -> Dict[str, Any]:
        """Wisp modules expose their public properties in a dictionary.
        The purpose of this method is to give an easy table of outwards facing attributes,
        for the purpose of logging, gui apps, etc.
        """
        return {
            "Output Dim": self.out_dim,
            "Num. Frequencies": self.num_freq,
            "Max Frequency": f"2^{self.max_freq_log2}",
            "Include Input": self.include_input,
        }


class ModulatedFourierFeatures(nn.Module):
    """WARNING: the code does not support non-graph inputs.
        It needs to be adapted for (batch, num_points, coordinates) format
        The FiLM Modulated Network with Fourier Embedding used for the experiments on Airfrans.
        The code relies on conditoning functions: film, film_linear and film_translate.
    Args:
        nn (_type_): _description_
    """

    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        num_frequencies=8,
        latent_dim=128,
        width=256,
        depth=3,
        modulate_scale=False,
        modulate_shift=True,
        frequency_embedding="nerf",
        include_input=True,
        scale=5,
        max_frequencies=32,
        base_frequency=1.25,
    ):
        super().__init__()
        self.frequency_embedding = frequency_embedding
        self.include_input = include_input
        if frequency_embedding == "nerf":
            self.embedding = NeRFEncoding(
                num_frequencies,
                max_frequencies,
                log_sampling=True,
                include_input=include_input,
                input_dim=input_dim,
                base_freq=base_frequency,
            )
            self.in_channels = [self.embedding.out_dim] + [width] * (depth - 1)

        elif frequency_embedding == "gaussian":
            self.scale = scale
            self.embedding = GaussianEncoding(
                embedding_size=num_frequencies * 2, scale=scale, dims=input_dim
            )
            embed_dim = (
                num_frequencies * 2 + input_dim
                if include_input
                else num_frequencies * 2
            )
            self.in_channels = [embed_dim] + [width] * (depth - 1)

        self.out_channels = [width] * (depth - 1) + [output_dim]
        self.latent_dim = latent_dim
        self.layers = nn.ModuleList(
            [nn.Linear(self.in_channels[k], self.out_channels[k]) for k in range(depth)]
        )
        self.depth = depth
        self.hidden_dim = width

        self.num_modulations = self.hidden_dim * (self.depth - 1)
        if modulate_scale and modulate_shift:
            # If we modulate both scale and shift, we have twice the number of
            # modulations at every layer and feature
            self.num_modulations *= 2
        self.latent_to_modulation = LatentToModulation(
            self.latent_dim, self.num_modulations, dim_hidden=256, num_layers=1
        )

        if modulate_shift and modulate_scale:
            self.conditioning = film
        elif modulate_scale and not modulate_shift:
            self.conditioning = film_linear
        else:
            self.conditioning = film_translate

    def modulated_forward(self, x, z):
        x_shape = x.shape[:-1]
        x = x.view(x.shape[0], -1, x.shape[-1])
        #print('x', x.shape, z.shape)
        features = self.latent_to_modulation(z)
        position = self.embedding(x)
        if self.frequency_embedding == "gaussian" and self.include_input:
            position = torch.cat([position, x], axis=-1)
        pre_out = self.conditioning(position, features, self.layers[:-1], torch.relu)
        out = self.layers[-1](pre_out)
        return out.view(*x_shape, out.shape[-1])

