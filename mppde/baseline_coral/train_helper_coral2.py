import torch
import random
from torch import nn, optim
from torch.utils.data import DataLoader
from mppde.baseline_coral.utils_coral import GraphCreator


def training_loop(model: torch.nn.Module,
                  unrolling: list,
                  n: int,
                  optimizer: torch.optim,
                  loader: DataLoader,
                  graph_creator: GraphCreator,
                  criterion: torch.nn.modules.loss,
                  device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    One training epoch with random starting points for every trajectory
    Args:
        model (torch.nn.Module): neural network PDE solver
        unrolling (list): list of different unrolling steps for each batch entry
        batch_size (int): batch size
        optimizer (torch.optim): optimizer used for training
        loader (DataLoader): training dataloader
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for traininfg
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: training losses
    """

    losses = 0
    for i, (u_super, _, x, idx) in enumerate(loader):
        # for (u_base, u_super, x, variables) in loader:
        batch_size = u_super.shape[0]
        variables = {}
        optimizer.zero_grad()

        #loss = 0

        # Randomly choose number of unrollings
        same_steps = [graph_creator.tw] * batch_size
        data, labels = graph_creator.create_data(u_super, same_steps)
        loss = 0
        if f'{model}' == 'GNN':
            graph = graph_creator.create_graph(
                data, labels, x[:, 0, :, :], variables, same_steps, mode='train').to(device)
            pred = model(graph)
            loss += criterion(pred, graph.y)

        for step in range(graph_creator.tw * 2, graph_creator.t_in - graph_creator.tw + 1, graph_creator.tw):
            same_steps = [step] * batch_size
            _, labels = graph_creator.create_data(u_super, same_steps)
            if f'{model}' == 'GNN':
                graph = graph_creator.create_next_graph(
                    graph, pred, labels, same_steps, mode='train').to(device)
                pred = model(graph)
            loss += criterion(pred, graph.y)

        loss.backward()
        losses += loss.item() * batch_size
        optimizer.step()
    losses = losses / n
    return losses


def test_unrolled_losses(model: torch.nn.Module,
                         steps: list,
                         batch_size: int,
                         nr_gt_steps: int,
                         ntest: int,
                         loader: DataLoader,
                         graph_creator: GraphCreator,
                         criterion: torch.nn.modules.loss,
                         device: torch.cuda.device = "cpu") -> torch.Tensor:
    """
    Loss for full trajectory unrolling, we report this loss in the paper
    Args:
        model (torch.nn.Module): neural network PDE solver
        steps (list): input list of possible starting (time) points
        nr_gt_steps (int): number of numerical input timesteps
        nx_base_resolution (int): spatial resolution of numerical baseline
        loader (DataLoader): dataloader [valid, test]
        graph_creator (GraphCreator): helper object to handle graph data
        criterion (torch.nn.modules.loss): criterion for training
        device (torch.cuda.device): device (cpu/gpu)
    Returns:
        torch.Tensor: valid/test losses
    """
    losses = []
    pred_inter_mse = 0
    pred_extra_mse = 0
    variables = {}
    for (u_super, _, x, idx) in loader:
        batch_size = u_super.shape[0]

        with torch.no_grad():
            same_steps = [graph_creator.tw * nr_gt_steps] * batch_size
            data, labels = graph_creator.create_data(u_super, same_steps)

            if f'{model}' == 'GNN':
                graph = graph_creator.create_graph(
                    data, labels, x[:, 0, :, :], variables, same_steps, mode='eval').to(device)
                pred = model(graph)
                loss = criterion(pred, graph.y)
            else:
                data, labels = data.to(device), labels.to(device)
                pred = model(data)
                loss = criterion(pred, labels)

            losses.append(loss.detach())

            # Unroll trajectory and add losses which are obtained for each unrolling
            for step in range(graph_creator.tw * (nr_gt_steps + 1), graph_creator.t_out - graph_creator.tw + 1, graph_creator.tw):
                same_steps = [step] * batch_size
                _, labels = graph_creator.create_data(u_super, same_steps)
                if f'{model}' == 'GNN':
                    graph = graph_creator.create_next_graph(
                        graph, pred, labels, same_steps, mode='eval').to(device)
                    pred = model(graph)
                    loss = criterion(pred, graph.y)
                else:
                    labels = labels.to(device)
                    pred = model(pred)
                    loss = criterion(pred, labels)
                losses.append(loss.detach())

        losses_inter = torch.stack(losses[:graph_creator.t_in])
        losses_extra = torch.stack(losses[graph_creator.t_in:])

        pred_inter_mse += torch.mean(losses_inter) * batch_size
        pred_extra_mse += torch.mean(losses_extra) * batch_size

    pred_inter_mse = pred_inter_mse / ntest
    pred_extra_mse = pred_extra_mse / ntest
    return pred_inter_mse, pred_extra_mse
