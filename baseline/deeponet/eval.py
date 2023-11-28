import math
import os
import shelve
import numpy as np
import torch
from torch import nn
from torch.nn import init
from coral.utils.data.dynamics_dataset import KEY_TO_INDEX
import einops

from deeponet import AR_forward


def eval_deeponet(deeponet, dataloader, device, timestamps_train, criterion, n_seq, n_frames_in, n_frames_out, detailed_mse, multichannel=False):
    """def eval_dino(
    dataloader,
    net_dyn,
    net_dec,
    device,
    method,
    criterion,
    state_dim,
    code_dim,
    coord_dim,
    detailed_mse,
    timestamps,
    n_seq,
    n_frames_train=0,
    n_frames_test=0,
    states_params=None,
    lr_adapt=0.0,
    n_steps=300,
    multichannel=False,
    save_best=True,
):"""
    """
    In_t: loss within train horizon.
    Out_t: loss outside train horizon.
    In_s: loss within observation grid.
    Out_s: loss outside observation grid.
    loss: loss averaged across in_t/out_t and in_s/out_s
    loss_in_t: loss averaged across in_s/out_s for in_t.
    loss_in_t_in_s, loss_in_t_out_s: loss in_t + in_s / out_s
    """

    (
        loss,
        loss_out_t,
        loss_in_t,
    ) = (0.0, 0.0, 0.0)

    set_requires_grad(deeponet, False)

    for j, (images, _, coords, idx) in enumerate(dataloader):
        # flatten spatial dims
        t = timestamps_train.to(device)
        ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
        model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')

        # permute axis for forward
        ground_truth = torch.permute(
            ground_truth, (0, 3, 1, 2)).to(device)  # [B, XY, C, T] -> [B, T, XY, C]
        model_input = torch.permute(
            model_input, (0, 3, 1, 2))[:, 0, :, :].to(device)  # ([B, XY, C, T] -> -> [B, T, XY, C] -> [B, XY, C]
        # On prend que la première grille (c'est tjs la mm dans deeponet) 
        b_size, t_size, hw_size, channels = ground_truth.shape

        # t is T, model_input is B, T, XY, grid, ground_truth is B, T, XY, C

        model_output = AR_forward(deeponet, t, model_input, ground_truth) 
        # B, T, XY, C
        print(model_output.shape)
        if n_frames_out == 0:
            loss += criterion(model_output, ground_truth).item() * b_size
        else : 
            loss = criterion(model_output, ground_truth).item() * b_size
            loss_in_t = criterion(model_output[:, :n_frames_in, :, :], ground_truth[:, :n_frames_in, :, :]).item() * b_size
            loss_out_t = criterion(model_output[:, n_frames_in:n_frames_in+n_frames_out, :, :], ground_truth[:, n_frames_in:n_frames_in+n_frames_out, :, :] ).item()* b_size
            
        if multichannel:
            detailed_mse.aggregate(model_output.detach(),
                                   ground_truth.detach())

    loss /= n_seq
    loss_in_t /= n_seq
    loss_out_t /= n_seq

    set_requires_grad(deeponet, True)

    return (
        loss,
        loss_in_t,
        loss_out_t,
    )


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf