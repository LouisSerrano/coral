import os
import random
import einops
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
from torch_geometric.data import Dataset as GeoDataset
from torch_geometric.data import Data

KEY_TO_INDEX = {
    "shallow-water-dino": {"height": 0, "vorticity": 1},
    "navier-stokes-dino": {"vorticity": 0},
    "mp-pde-burgers": {"vorticity": 0},
}


class GraphTemporalDataset(GeoDataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid):
        super().__init__(None, None, None)
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

        self.v = einops.rearrange(v, 'b ... c t -> b (...) c t')
        self.c = einops.rearrange(grid, 'b ... c t -> b (...) c t')

        # print('self values', self.v.shape)
        # print('self grid', self.c.shape)

        self.T = T

    def len(self):
        return len(self.v)

    def get(self, idx):
        """The tempral dataset returns whole trajectories, identified by the index.

        Args:
            idx (int): idx of the trajectory

        Returns:
            sample_v (torch.Tensor): the trajectory with shape (Dx Dy C T)
            sample_z (torch.Tensor): the codes with shape (L T)
            sample_c (torch.Tensor): the spatial coordinates (Dx Dy 2)
        """
        graph = Data(pos=self.c[0, :, :, 0].clone())
        graph.pos_t = torch.arange(self.T)  # 20
        graph.images = self.v[idx].clone()  # 204, 1, 20

        # print("graph.pos_t.shape : ", graph.pos_t.shape)
        # print("graph.images.shape : ", graph.images.shape) # # 204, 1, 20
        # print("self.c.shape : ", self.c.shape) # [256, 204, 2, 20]
        # print("self.v.shape : ", self.v.shape) # [256, 204, 1, 20]

        return graph, idx
