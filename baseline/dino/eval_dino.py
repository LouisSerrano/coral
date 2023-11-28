import numpy as np
import einops
import torch
from torch import nn
from torch.nn import init
from coral.utils.data.dynamics_dataset import KEY_TO_INDEX
from torchdiffeq import odeint
from utils import set_requires_grad

def eval_dino(
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
):
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
    set_requires_grad(net_dec, False)
    set_requires_grad(net_dyn, False)
    for j, (images, _, coords, idx) in enumerate(dataloader):
        ground_truth = images.to(device)
        if multichannel:
            model_input = coords.unsqueeze(-2)
            model_input = einops.repeat(model_input, '... c s -> ... (c k) s', k = 2).to(device)
        else:
            model_input = coords.unsqueeze(-2).to(device)
        index = idx.to(device)
        b_size, t_size = ground_truth.shape[0], ground_truth.shape[1]
        if lr_adapt != 0.0:
            loss_min_test = 1e30
            states_params_out = nn.ParameterList(
                [
                    nn.Parameter(torch.zeros(1, code_dim * state_dim).to(device))
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
                model_output, _ = net_dec(model_input[:, 0:1], states)
                loss_l2 = criterion(
                    model_output[:, :, ...], ground_truth[:, 0:1, ...]
                )
                if loss_l2 < loss_min_test and save_best:
                    loss_min_test = loss_l2.item()
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

            if n_frames_test != 0:
                loss_in_t += criterion(
                    model_output[:, :n_frames_train, ...],
                    ground_truth[:, :n_frames_train, ...],
                ).item() * b_size
                loss += criterion(model_output, ground_truth).item() * b_size
                loss_out_t += criterion(
                    model_output[:, n_frames_train:, ...],
                    ground_truth[:, n_frames_train:, ...],
                ).item() * b_size
            if n_frames_test == 0:
                loss += criterion(model_output, ground_truth).item() * b_size
        if multichannel:
            detailed_mse.aggregate(model_output.detach(), ground_truth.detach())
    loss /= n_seq
    loss_in_t /= n_seq
    loss_out_t /= n_seq
    set_requires_grad(net_dec, True)
    set_requires_grad(net_dyn, True)
    return (
        loss,
        loss_in_t,
        loss_out_t,
        detailed_mse
    )

class DetailedMSE():
    def __init__(self, keys, dataset_name="shallow-water-dino", mode="train", n_trajectories=256):
        self.keys = keys
        self.mode = mode
        self.dataset_name = dataset_name
        self.n_trajectories = n_trajectories
        self.reset_dic()

    def reset_dic(self):
        dic = {}
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] = 0
        self.dic = dic

    def aggregate(self, u_pred, u_true):
        n_samples = u_pred.shape[0]
        for key in self.keys:
            idx = KEY_TO_INDEX[self.dataset_name][key]
            self.dic[f"{key}_{self.mode}_mse"] += (
                (u_pred[..., idx] - u_true[..., idx])**2).mean()*n_samples

    def get_dic(self):
        dic = self.dic
        for key in self.keys:
            dic[f"{key}_{self.mode}_mse"] /= self.n_trajectories
        return self.dic