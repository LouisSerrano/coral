import torch
import torch.nn as nn
import time
import hydra
import wandb
import einops
from pathlib import Path
import os
from torchdiffeq import odeint

from coral.metalearning import outer_step
from coral.utils.models.scheduling import ode_scheduling
from coral.utils.models.get_inr_reconstructions import get_reconstructions


def forward_dino(net_dec, net_dyn, batch, n_seq, states_params, code_dim, n_steps, lr_adapt, device, criterion, timestamps, save_best=True, method="rk4"):
    (images, _, coords, idx) = batch
    set_requires_grad(net_dec, False)
    set_requires_grad(net_dyn, False)
    # print("coords.shape, images.shape : ", coords.shape, images.shape)

    ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
    model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')
    # print("ground_truth.shape, model_input.shape : ", ground_truth.shape, model_input.shape)

    ground_truth = torch.permute(
        ground_truth, (0, 3, 1, 2)).to(device)  # [B, XY, C, T] -> [B, T, XY, C]
    model_input = torch.permute(
        model_input, (0, 3, 1, 2)).to(device)  # ([B, XY, C, T] -> -> [B, T, XY, C]
    # print("ground_truth.shape, model_input.shape : ", ground_truth.shape, model_input.shape)

    b_size, t_size, hw_size, state_dim = ground_truth.shape
    _, _, _, coord_dim = model_input.shape

    model_input = model_input.view(
        b_size, t_size, hw_size, 1, coord_dim
    ).expand(b_size, t_size, hw_size, state_dim, coord_dim)  # B, T, XY, C, grid

    # n_seq = b_size
    loss_min_test = 1e30
    index = idx.to(device)
    states_params_out = nn.ParameterList(
        [
            nn.Parameter(torch.zeros(
                1, code_dim * state_dim).to(device))
            for _ in range(n_seq)
        ]
    )

    optim_states_out = torch.optim.Adam(states_params_out, lr=lr_adapt)
    for i in range(n_steps):
        states_params_index = [states_params_out[d] for d in index]
        states_params_index = torch.stack(states_params_index, dim=1)
        states = states_params_index.permute(1, 0, 2).view(
            b_size, 1, state_dim, code_dim
        )
        model_output, _ = net_dec(model_input[:, 0:1, ...], states)
        loss_l2 = criterion(
            model_output[:, :, ...], ground_truth[:, 0:1, ...]
        )
        if loss_l2 < loss_min_test and save_best:
            loss_min_test = loss_l2
            best_states_params_index = states_params_index
        loss_opt_new = loss_l2
        loss_opt = loss_opt_new
        optim_states_out.zero_grad(True)
        loss_opt.backward()
        optim_states_out.step()
    if save_best:
        states_params_index = best_states_params_index
    with torch.no_grad():
        if lr_adapt == 0.0:
            states_params_index = [states_params[d] for d in index]
            states_params_index = torch.stack(states_params_index, dim=1)
        codes = odeint(
            net_dyn, states_params_index[0], timestamps, method=method
        )  # t x batch x dim
        codes = codes.permute(1, 0, 2).view(
            b_size, t_size, state_dim, code_dim
        )  # batch x t x dim

        model_output, _ = net_dec(model_input, codes)

    return einops.rearrange(model_output, 'B T X C -> B X C T')


def forward_coral(inr, model, batch, inner_steps, alpha, use_rel_loss, timestamps, z_mean, z_std, dataset_name):
    (images, modulations, coords, idx) = batch

    inr.eval().cuda()
    images = images.cuda()
    modulations = modulations.cuda()
    coords = coords.cuda()
    n_samples, xy_size, channel, t_size = images.shape

    images = einops.rearrange(images, "b ... t -> (b t) ...").cuda()
    modulations = einops.rearrange(modulations, "b ... t -> (b t) ...")
    coords = einops.rearrange(coords, "b ... t -> (b t) ...").cuda()
    # [40, 1024, 2]) = (b*t, XY, grid)

    outputs = outer_step(
        inr,
        coords,
        images,
        inner_steps,
        alpha.cuda(),
        is_train=False,
        return_reconstructions=False,
        gradient_checkpointing=False,
        use_rel_loss=use_rel_loss,
        loss_type="mse",
        modulations=torch.zeros_like(modulations).cuda(),
    )

    (images, _, coords, idx) = batch
    model.eval().cuda()
    images = images.cuda()

    pred_modulations = einops.rearrange(
        outputs['modulations'], "(b t) ... -> b ... t", t=t_size)
    modulations = pred_modulations  # Take modulations from inr encoding

    modulations = (modulations - z_mean.cuda()) / z_std.cuda()

    coords = coords.cuda()
    n_samples = images.shape[0]

    z_pred = ode_scheduling(
        odeint, model, modulations, timestamps, 0)

    pred = get_reconstructions(
        inr, coords, z_pred, z_mean, z_std, dataset_name
    )

    return pred


def set_requires_grad(module, tf=False):
    module.requires_grad = tf
    for param in module.parameters():
        param.requires_grad = tf
