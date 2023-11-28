import numpy as np
import torch
import einops
import random
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeoDataset
from torch_cluster import knn_graph

KEY_TO_INDEX = {"shallow-water-dino": 
                {"height": 0, "vorticity": 1},
                "navier-stokes-dino":
                {"vorticity": 0}
                }

def mppde_create_data(datapoints: torch.Tensor, batch: torch.Tensor, pos: torch.Tensor, steps: list, tw: int =1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Getting data for PDE training at different time points
        Args:
            datapoints (torch.Tensor): trajectory
            steps (list): list of different starting points for each batch entry
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: input data and label
        """
        data = torch.Tensor().cuda()
        labels = torch.Tensor().cuda()
        out_pos = torch.Tensor().cuda()
        out_batch = torch.Tensor().cuda()
        for k, step in enumerate(steps):
            dp = datapoints[batch == k]
            d = dp[..., step - tw:step].squeeze(-1)
            l = dp[..., step:tw + step].squeeze(-1)
            data = torch.cat((data, d), 0)
            labels = torch.cat((labels, l), 0)
            out_pos = torch.cat((out_pos, pos[batch==k]), 0)
            out_batch = torch.cat((out_batch, batch[batch==k]), 0)
        
        #data = einops.rearrange(data, 'b l s -> b (l s)')
        #labels = einops.rearrange(labels, 'b l s -> b (l s)')
        new_graph = Data(pos = out_pos, input = data, labels = labels, batch = out_batch) #.cuda()
        return new_graph

def mppde_pushforward(model, graph, bundle_size=1):
    # (B, L, T) ->
    #true_codes = einops.rearrange(true_codes, 'b l (t s) -> b (l s) t', s=bundle_size)
    #pred_codes = torch.zeros_like(true_codes)
    T = graph.images.shape[-1]
    batch_size = len(graph)
    unrolling = [0, 1]
    unrolled_graphs = random.choice(unrolling)
    steps = [t for t in range(bundle_size, T - bundle_size - (bundle_size * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
    random_steps = random.choices(steps, k=batch_size)

    print('data before', graph.images.shape)

    new_graph = mppde_create_data(graph.images, graph.batch, graph.pos, random_steps, bundle_size)
    new_graph.input = torch.cat((new_graph.pos, new_graph.input), -1)
    new_graph.batch = new_graph.batch.long()
    labels = new_graph.labels


    #radius = 0.25
    max_neighbours = 8

    #new_graph.edge_index = radius_graph(new_graph.pos,
    #                                radius,
    #                                new_graph.batch,
    #                                loop=False,
    #                                max_num_neighbors=max_neighbours,)
    new_graph.pos = new_graph.pos.cpu()
    new_graph.batch = new_graph.batch.cpu()

    new_graph.edge_index = knn_graph(new_graph.pos,
                                    max_neighbours,
                                    new_graph.batch,
                                    loop=False)
    new_graph.edge_index = new_graph.edge_index.cuda()
    new_graph.pos = new_graph.pos.cuda()
    new_graph.batch = new_graph.batch.cuda()
    
    with torch.no_grad():
        for k in range(unrolled_graphs):
            random_steps = [rs + bundle_size + k for rs in random_steps]
            labels_graph = mppde_create_data(graph.images, graph.batch, graph.pos, random_steps, bundle_size)
            pred = model(new_graph)
            new_graph.input = torch.cat((new_graph.pos, pred), -1)
            new_graph.batch = new_graph.batch.long()
            labels = labels_graph.labels

    pred = model(new_graph)
    loss = ((pred-labels)**2).mean()
    return loss

def mppde_test_rollout(model, graph, bundle_size=1):
    # (B, L, T) ->
    #true_codes = einops.rearrange(true_codes, 'b l (t s) -> b (l s) t', s=bundle_size)
    #pred_codes = torch.zeros_like(true_codes)
    T = graph.images.shape[-1]

    u_pred = torch.zeros_like(graph.images)
    u_pred[..., 0] = graph.images[..., 0]

    graph.input = torch.cat((graph.pos, graph.images[..., 0]), -1)

    #radius = 0.25
    max_neighbours = 8

    #graph.edge_index = radius_graph(graph.pos,
    #                                radius,
    #                                graph.batch,
    #                                loop=False,
    #                                max_num_neighbors=max_neighbours,)
    graph.pos = graph.pos.cpu()
    graph.batch = graph.batch.cpu()
    graph.edge_index = knn_graph(graph.pos,
                                    max_neighbours,
                                    graph.batch,
                                    loop=False)
    graph.edge_index = graph.edge_index.cuda()
    graph.pos = graph.pos.cuda()
    graph.batch = graph.batch.cuda()

    #inpt = einops.rearrange(inpt, 'b l s -> b (l s)')

    with torch.no_grad():
        for step in range(1, T):
            pred = model(graph)
            u_pred[..., step] = pred # tmp
            graph.input = torch.cat((graph.pos, pred), -1)

    return u_pred

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

        print('self values', self.v.shape)
        print('self grid', self.c.shape)

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
        graph.pos_t = torch.arange(self.T)
        graph.images = self.v[idx].clone()

        #print('toto pos', graph.pos.shape)
        #print('toto images', graph.images.shape)
        #graph.input = torch.cat((graph.pos, graph.images), -1)

        return graph, idx