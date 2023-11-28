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
import xarray as xr
from scipy import io
from torch.utils.data import Dataset


class OperatorDataset(Dataset):
    """Custom dataset for encoding task. Iterates over the input/output values, grids, and codes."""

    def __init__(
        self,
        a,
        u,
        grid,
        latent_dim_a=64,
        latent_dim_u=64,
        dataset_name=None,
        data_to_encode=None,
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        self.a = a
        self.u = u
        self.grid = grid  # repeat_coordinates(grid_a, a.shape[0])
        self.latent_dim_a = latent_dim_a
        self.latent_dim_u = latent_dim_u
        self.dataset_name = dataset_name
        self.data_to_encode = data_to_encode

        self.z_a = torch.zeros((self.a.shape[0], latent_dim_a))
        self.z_u = torch.zeros((self.u.shape[0], latent_dim_u))

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_a = self.a[idx, ...]
        sample_u = self.u[idx, ...]
        sample_z_a = self.z_a[idx, ...]
        sample_z_u = self.z_u[idx, ...]
        sample_grid = self.grid[idx, ...]

        return (
            sample_a,
            sample_u,
            sample_z_a,
            sample_z_u,
            sample_grid,
            idx,
        )

    def __setitem__(self, new_z_a, new_z_u, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z_a[idx, ...] = new_z_a
        self.z_u[idx, ...] = new_z_u


class ConvOperatorDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        a,
        u,
        grid,
        grid_size=8,
        latent_dim_a=64,
        latent_dim_u=64,
        dataset_name=None,
        data_to_encode=None,
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        self.a = a
        self.u = u
        self.grid = grid  # repeat_coordinates(grid_a, a.shape[0])
        self.latent_dim_a = latent_dim_a
        self.latent_dim_u = latent_dim_u
        self.dataset_name = dataset_name
        self.data_to_encode = data_to_encode

        self.z_a = torch.zeros(
            (self.a.shape[0], latent_dim_a, *([grid_size] * grid.shape[-1]))
        )
        self.z_u = torch.zeros(
            (self.u.shape[0], latent_dim_u, *([grid_size] * grid.shape[-1]))
        )

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_a = self.a[idx, ...]
        sample_u = self.u[idx, ...]
        sample_z_a = self.z_a[idx, ...]
        sample_z_u = self.z_u[idx, ...]
        sample_grid = self.grid[idx, ...]

        return (
            sample_a,
            sample_u,
            sample_z_a,
            sample_z_u,
            sample_grid,
            idx,
        )

    def __setitem__(self, new_z_a, new_z_u, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z_a[idx, ...] = new_z_a
        self.z_u[idx, ...] = new_z_u


class DatasetWithCode(Dataset):
    """A simple dataset with code. No input output"""

    def __init__(self, v, grid, latent_dim=64, dataset_name=None, data_to_encode=None):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """
        self.v = v
        self.z = torch.zeros((v.shape[0], latent_dim))
        self.c = grid
        self.latent_dim = latent_dim
        self.dataset_name = dataset_name
        self.data_to_encode = data_to_encode

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class ConvDatasetWithCode(Dataset):
    """A simple dataset with code. The codes have spatial dimensions as well."""

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
            grid_size (int): Spatial dimensions of the code.
            latent_dim (int, optional): Latent dimension of the code. Defaults to 32.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """

        self.v = v
        self.z = torch.zeros(
            (self.v.shape[0], latent_dim, *([grid_size] * grid.shape[-1]))
        )
        self.c = grid
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.dataset_name = dataset_name
        self.data_to_encode = data_to_encode

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values
