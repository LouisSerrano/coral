import torch
import torch.nn as nn
import time
import hydra
import wandb
import einops
from pathlib import Path
import os
from torchdiffeq import odeint

from torch_geometric.nn import radius_graph



def forward_mppde(model, batch, graph_creator, nr_gt_steps, device):

    (u_super, _, x, idx) = batch

    # u_super = einops.rearrange(u_super, 'B X C T -> B T X C')
    # x = einops.rearrange(x, 'B X C T -> B T X C')

    batch_size = u_super.shape[0]
    preds = u_super[:, [0], :, :].to(device)
    variables = {}

    with torch.no_grad():
        same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
        data, labels = graph_creator.create_data(u_super, same_steps)

        graph = graph_creator.create_graph(
            data, labels, x[:, 0, :, :], variables, same_steps, mode='eval').to(device)
        pred = model(graph)

        preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), -1)

        # Unroll trajectory and add losses which are obtained for each unrolling
        # Unroll trajectory and add losses which are obtained for each unrolling
        for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_out - graph_creator.tw + 1, graph_creator.tw):
            same_steps = [step] * batch_size
            _, labels = graph_creator.create_data(u_super, same_steps)

            graph = graph_creator.create_next_graph(
                graph, pred, labels, same_steps, mode='eval').to(device)
            pred = model(graph)
            preds = torch.cat((preds, pred.unsqueeze(0).unsqueeze(0)), -1)

    return einops.rearrange(preds, 'B C X T -> B T X C')


def forward_mppde_louis(model, batch):
    (graph, idx) = batch
    graph = graph.cuda()

    T = graph.images.shape[-1]

    u_pred = torch.zeros_like(graph.images)
    u_pred[..., 0] = graph.images[..., 0]

    graph.input = torch.cat((graph.pos, graph.images[..., 0]), -1)

    radius = 0.25
    max_neighbours = 8

    graph.edge_index = radius_graph(graph.pos,
                                    radius,
                                    graph.batch,
                                    loop=False,
                                    max_num_neighbors=max_neighbours,)
    with torch.no_grad():
        for step in range(1, T):
            pred = model(graph)
            u_pred[..., step] = pred  # tmp
            graph.input = torch.cat((graph.pos, pred), -1)

    return u_pred
