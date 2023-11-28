import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.nn import functional as F
from typing import Tuple
from torch_geometric.data import Data
from torch_cluster import radius_graph, knn_graph


class PDE_CORAL():
    def __init__(self, pos_dim, input_dim, nt, nx, tmin, t_in, t_out, dt, grid_type, batch_size) -> None:
        self.pos_dim = pos_dim
        self.input_dim = input_dim
        self.grid_size = (nt, nx)
        self.tmin = tmin
        self.t_in = t_in
        self.t_out = t_out
        self.grid_type = grid_type
        self.dt = dt
        self.L = 1
        self.batch_size = batch_size
        self.nx = nx
        self.nt = nt


class GraphCreator(nn.Module):
    def __init__(self,
                 pde: PDE_CORAL,
                 neighbors: int = 2,
                 time_window: int = 5,
                 x_resolution: int = 100
                 ) -> None:
        """
        Initialize GraphCreator class
        Args:
            pde (PDE): PDE at hand [CE, WE, ...]
            neighbors (int): how many neighbors the graph has in each direction
            time_window (int): how many time steps are used for PDE prediction
            time_ration (int): temporal ratio between base and super resolution
            space_ration (int): spatial ratio between base and super resolution
        Returns:
            None
        """
        super().__init__()
        self.pde = pde
        self.n = neighbors
        self.tw = time_window
        self.t_in = self.pde.t_in
        self.t_out = self.pde.t_out
        self.x_res = x_resolution

        print("self.t_in, self.t_out, self.tw, self.x_res : ",
              self.t_in, self.t_out, self.tw, self.x_res)

        assert isinstance(self.n, int)
        assert isinstance(self.tw, int)

    def create_data(self, datapoints: torch.Tensor, steps: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor()
        labels = torch.Tensor()
        for (dp, step) in zip(datapoints, steps):
            d = dp[step - self.tw:step]
            l = dp[step:self.tw + step]
            data = torch.cat((data, d), 0)
            labels = torch.cat((labels, l), 0)
        return data, labels

    def create_graph(self,
                     data: torch.Tensor,
                     labels: torch.Tensor,
                     x: torch.Tensor,
                     variables: dict,
                     steps: list,
                     mode: str) -> Data:
        """
        Getting graph structure out of data sample
        previous timesteps are combined in one node
        Args:
            data (torch.Tensor): input data tensor
            labels (torch.Tensor): label tensor
            x (torch.Tensor): spatial coordinates tensor
            variables (dict): dictionary of equation specific parameters
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        nx = x[0].shape[0]

        if mode == 'train':
            t = torch.linspace(self.pde.tmin, self.pde.t_in, self.pde.t_in)
        if mode == 'eval':
            t = torch.linspace(self.pde.tmin, self.pde.t_out, self.pde.t_out)

        u, x_pos, t_pos, y, batch = torch.Tensor(), torch.Tensor(
        ), torch.Tensor(), torch.Tensor(), torch.Tensor()
        for b, (data_batch, labels_batch, step) in enumerate(zip(data, labels, steps)):
            u = torch.cat(
                (u, torch.cat([d.unsqueeze(-1) for d in data_batch])), )
            y = torch.cat(
                (y, torch.cat([l.unsqueeze(-1) for l in labels_batch])), )
            x_pos = torch.cat((x_pos, x[0]), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
            batch = torch.cat((batch, torch.ones(nx) * b), )

        # Calculate the edge_index
        if f'{self.pde.grid_type}' == 'regular':
            dx = x[0][1] - x[0][0]
            radius = self.n * dx + 0.0001
            radius = radius.mean()
            edge_index = radius_graph(
                x_pos, r=radius, batch=batch.long(), loop=False)

        elif f'{self.pde.grid_type}' == 'irregular':
            edge_index = knn_graph(
                x_pos, k=self.n, batch=batch.long(), loop=False)

        graph = Data(x=u, edge_index=edge_index)
        graph.y = y

        if mode == 'train':
            t_pos = t_pos / self.pde.t_in
        if mode == 'eval':
            t_pos = t_pos / self.pde.t_out

        if self.pde.pos_dim == 3:
            # TODO verify with team
            graph.pos = torch.cat(
                (t_pos[:, None], x_pos[:, 0:1], x_pos[:, 1:2], x_pos[:, 2:3]), 1)
        if self.pde.pos_dim == 2:
            # TODO verify with team
            graph.pos = torch.cat(
                (t_pos[:, None], x_pos[:, 0:1], x_pos[:, 1:2]), 1)
        if self.pde.pos_dim == 1:
            # TODO verify with team
            graph.pos = torch.cat((t_pos[:, None], x_pos[:, 0:1]), 1)

        graph.batch = batch.long()
        return graph

    def create_next_graph(self,
                          graph: Data,
                          pred: torch.Tensor,
                          labels: torch.Tensor,
                          steps: list,
                          mode: str) -> Data:
        """
        Getting new graph for the next timestep
        Method is used for unrolling and when applying the pushforward trick during training
        Args:
            graph (Data): Pytorch geometric data object
            pred (torch.Tensor): prediction of previous timestep ->  input to next timestep
            labels (torch.Tensor): labels of previous timestep
            steps (list): list of different starting points for each batch entry
        Returns:
            Data: Pytorch Geometric data graph
        """
        # Output is the new input
        graph.x = torch.cat((graph.x, pred), 1)[:, self.tw:]
        nx = self.pde.nx

        if mode == 'train':
            t = torch.linspace(self.pde.tmin, self.pde.t_in, self.pde.t_in)
        if mode == 'eval':
            t = torch.linspace(self.pde.tmin, self.pde.t_out, self.pde.t_out)

        # Update labels and input timesteps
        y, t_pos = torch.Tensor(), torch.Tensor()
        for (labels_batch, step) in zip(labels, steps):
            y = torch.cat(
                (y, torch.cat([l.unsqueeze(-1) for l in labels_batch])), )
            t_pos = torch.cat((t_pos, torch.ones(nx) * t[step]), )
        graph.y = y

        if mode == 'train':
            t_pos = t_pos / self.pde.t_in
        if mode == 'eval':
            t_pos = t_pos / self.pde.t_out

        graph.pos[:, 0] = t_pos

        return graph


KEY_TO_INDEX = {"shallow-water-dino":
                {"height": 0, "vorticity": 1},
                "navier-stokes-dino":
                {"vorticity": 0}
                }


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
        T = self.v.shape[1]

        self.index_value = None
        if (data_to_encode is not None) and (dataset_name is not None):
            self.index_value = KEY_TO_INDEX[dataset_name][data_to_encode]
            self.z = torch.zeros((N, T, self.latent_dim))
            self.output_dim = 1

        if data_to_encode is None:
            c = len(KEY_TO_INDEX[dataset_name].keys())
            # one code for the height / vorticity
            # if c == 1, we squeeze it
            self.z = torch.zeros((N, T, self.latent_dim, c)).squeeze()

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
            sample_v = self.v[idx, ..., self.index_value:self.index_value+1]
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
