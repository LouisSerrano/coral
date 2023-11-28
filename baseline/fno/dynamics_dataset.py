import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
from scipy import io
from torch.utils.data import Dataset

from coral.utils.load_data import repeat_coordinates

KEY_TO_INDEX = {"shallow-water-dino": 
                {"height": 0, "vorticity": 1},
                "navier-stokes-dino":
                {"vorticity": 0}
                }

def rearrange(set):
    set.v = einops.rearrange(set.v, 'N ... T -> (N T) ... 1')
    set.z = einops.rearrange(set.z, 'N ... T -> (N T) ... ')
    set.c = einops.rearrange(set.c, 'N ... T -> (N T) ... ')
    return set

class TemporalDatasetWithCode(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim=64, dataset_name=None, data_to_encode=None):
        """
        Args:
            v (torch.Tensor): Dataset values, with shape (N Dx Dy C T). Where N is the
            number of trajectories, Dx the size of the first spatial dimension, Dy the size
            of the second spatial dimension, C the number of channels (ususally 1), and T the
            number of timestamps.
            grid (torch.Tensor): Coordinates, with shape (N Dx Dy 2). We suppose that we have
            same grid over time.
            latent_dim (int, optional): Latent dimension of the code. Defaults to 64.
        """
        N = v.shape[0]
        T = v.shape[-1]
        self.v = v
        self.c = grid  # repeat_coordinates(grid, N).clone()
        self.output_dim = self.v.shape[-2]
        self.input_dim = self.c.shape[-2]
        self.z = torch.zeros((N, latent_dim, T))
        self.latent_dim = latent_dim
        self.T = T
        self.dataset_name = dataset_name
        self.set_data_to_encode(data_to_encode)

    def set_data_to_encode(self, data_to_encode):
        self.data_to_encode = data_to_encode
        dataset_name = self.dataset_name
        N = self.v.shape[0]
        T = self.v.shape[-1]

        self.index_value = None
        if (data_to_encode is not None) and (dataset_name is not None):
            self.index_value = KEY_TO_INDEX[dataset_name][data_to_encode]
            self.z = torch.zeros((N, self.latent_dim, T))
            self.output_dim = 1

        if data_to_encode is None:
            c = len(KEY_TO_INDEX[dataset_name].keys())
            # one code for the height / vorticity
            # if c == 1, we squeeze it
            self.z = torch.zeros((N, self.latent_dim, c, T)).squeeze()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.index_value is not None:
            sample_v = self.v[idx, ..., self.index_value:self.index_value+1, :]
        else:
            sample_v = self.v[idx, ...]

        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        """How to save efficiently the updated codes.

        Args:
            z_values (torch.Tensor): the updated latent code for the whole trajectory.
            idx (int): idx of the trajectory.
        """
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class ConvTemporalDatasetWithCode(Dataset):
    """Custom dataset for encoding task with ConvINR. The codes hava now spatial dimensions as well."""

    def __init__(
        self,
        v,
        grid,
        grid_size=8,
        latent_dim=32,
        dataset_name=None,
        data_to_encode=None,
    ):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            grid_size (int): the spatial discretization for the code. Defaults to 8
            latent_dim (int, optional): Latent dimension of the code. Defaults to 32.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """

        N = v.shape[0]
        T = v.shape[-1]
        self.z = torch.zeros((N, latent_dim, *([grid_size] * grid.shape[-1]), T))
        self.c = grid  # repeat_coordinates(grid, v.shape[0]).clone()
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.T = T
        self.dataset_name = dataset_name
        self.data_to_encode = data_to_encode
        self.index_value = None
        if (data_to_encode is not None) and (dataset_name is not None):
            for dataset_type in KEY_TO_INDEX.keys():
                if dataset_type in dataset_name:
                    self.index_value = KEY_TO_INDEX[dataset_type][data_to_encode]
                    break

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.index_value is not None:
            sample_v = self.v[idx, ..., self.index_value, :]
        else:
            sample_v = self.v[idx, ...]

        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        z_values = z_values.clone()
        self.z[idx, ...] = z_values
