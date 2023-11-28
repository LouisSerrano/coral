import numpy as np
import torch
import einops
import random
from typing import Tuple
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph

def ode_scheduling(_int, _f, true_codes, t, epsilon, method="rk4"):
    if epsilon < 1e-3:
        epsilon = 0
    if epsilon == 0:
        codes = _int(_f, y0=true_codes[..., 0], t=t, method=method)
    else:
        eval_points = np.random.random(len(t)) < epsilon
        eval_points[-1] = False
        eval_points = eval_points[1:]

        start_i, end_i = 0, None
        codes = []
        for i, eval_point in enumerate(eval_points):
            if eval_point == True:
                end_i = i + 1
                t_seg = t[start_i: end_i + 1]
                res_seg = _int(
                    _f, y0=true_codes[..., start_i], t=t_seg, method=method)

                if len(codes) == 0:
                    codes.append(res_seg)
                else:
                    codes.append(res_seg[1:])
                start_i = end_i
        t_seg = t[start_i:]
        res_seg = _int(_f, y0=true_codes[..., start_i], t=t_seg, method=method)
        if len(codes) == 0:
            codes.append(res_seg)
        else:
            codes.append(res_seg[1:])
        codes = torch.cat(codes, dim=0)
    # (t b l) -> (b l t)
    return torch.movedim(codes, 0, -1)


def resnet_scheduling(model, true_codes, epsilon, bundle_size=25, index_start=0):
    # (B, L, T) ->
    true_codes = einops.rearrange(true_codes, 'b l (t s) -> b (l s) t', s=bundle_size)
    pred_codes = torch.zeros_like(true_codes)

    if epsilon < 1e-3:
        epsilon = 0
    if epsilon == 0:
        inpt = true_codes[..., index_start]
        pred_codes[..., :index_start] = true_codes[..., :index_start]

        for step in range(index_start+1, true_codes.shape[-1]):
            pred = model(inpt)
            pred_codes[..., step] = pred
            inpt = pred
    else:
        inpt = true_codes[..., index_start]
        pred_codes[..., :index_start] = true_codes[..., :index_start]

        for step in range(index_start+1, true_codes.shape[-1]):
            pred = model(inpt)
            pred_codes[..., step] = pred
            if np.random.uniform() < epsilon:
                inpt = true_codes[..., step]
            else:
                inpt = pred
    # (t b l) -> (b l t)
    return einops.rearrange(pred_codes, 'b (l s) t -> b l (s t)', s=bundle_size)

def resnet_pushforward(model, true_codes, bundle_size=25):
    # (B, L, T) ->
    #true_codes = einops.rearrange(true_codes, 'b l (t s) -> b (l s) t', s=bundle_size)
    #pred_codes = torch.zeros_like(true_codes)
    T = true_codes.shape[-1]
    batch_size = true_codes.shape[0]
    unrolling = [0, 1]
    unrolled_graphs = random.choice(unrolling)
    steps = [t for t in range(bundle_size, T - bundle_size - (bundle_size * unrolled_graphs) + 1)]
        # Randomly choose starting (time) point at the PDE solution manifold
    random_steps = random.choices(steps, k=batch_size)
    data, labels = create_data(true_codes, random_steps, bundle_size)

    with torch.no_grad():
        for _ in range(unrolled_graphs):
            random_steps = [rs + bundle_size for rs in random_steps]
            _, labels = create_data(true_codes, random_steps, bundle_size)
            data = model(data)
            labels = labels.cuda()

    pred = model(data)
    loss = ((pred-labels)**2).mean()
    return loss

def test_rollout(model, true_codes, bundle_size=25, index_start=0):
    # (B, L, T) ->
    #true_codes = einops.rearrange(true_codes, 'b l (t s) -> b (l s) t', s=bundle_size)
    #pred_codes = torch.zeros_like(true_codes)
    pred_codes = torch.zeros_like(true_codes)
    pred_codes[..., :bundle_size*(index_start+1)] = true_codes[..., :bundle_size*(index_start+1)]

    T = true_codes.shape[-1]
    num_steps = (T - bundle_size*(index_start+1))//bundle_size
    inpt = true_codes[..., bundle_size*index_start: bundle_size*(index_start+1)]
    #inpt = einops.rearrange(inpt, 'b l s -> b (l s)')

    for step in range(index_start + 1, index_start + 1 + num_steps):
        inpt = model(inpt)
        #tmp = einops.rearrange(inpt, 'b (l s) -> b l s', s=bundle_size)
        pred_codes[..., bundle_size*step: bundle_size*(step+1)] = inpt # tmp

    return pred_codes

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
        new_graph = Data(pos = out_pos, input = data, labels = labels, batch = out_batch).cuda()
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

    new_graph.edge_index = knn_graph(new_graph.pos,
                                    max_neighbours,
                                    new_graph.batch,
                                    loop=False)

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

    graph.edge_index = knn_graph(graph.pos,
                                    max_neighbours,
                                    graph.batch,
                                    loop=False)

    #inpt = einops.rearrange(inpt, 'b l s -> b (l s)')

    with torch.no_grad():
        for step in range(1, T):
            pred = model(graph)
            u_pred[..., step] = pred # tmp
            graph.input = torch.cat((graph.pos, pred), -1)

    return u_pred
