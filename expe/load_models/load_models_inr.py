import torch
import torch.nn as nn
import time
import hydra
import wandb
import einops
from pathlib import Path
import os

from coral.utils.data.load_modulations import load_dynamics_modulations
from coral.utils.models.load_inr import load_inr_model
from coral.mlp import Derivative as Derivative_coral
from dino.ode_model import Decoder
from dino.ode_model import Derivative as Derivative_dino


def load_coral(root_dir, inr_run_name, dyn_run_name, data_to_encode, input_dim, output_dim, trainset, testset, multichannel, code_dim_coral, width_dyn_coral, depth_dyn_coral, inner_steps, return_mod=False):
    # TODO : load parameters from config
    inr, alpha = load_inr_model(
        root_dir / "inr",
        inr_run_name,
        data_to_encode,
        input_dim=input_dim,
        output_dim=output_dim,
    )

    modulations = load_dynamics_modulations(
        trainset,
        testset,
        inr,
        root_dir / "modulations",
        inr_run_name,
        inner_steps=inner_steps,
        alpha=alpha,
        batch_size=2,
        data_to_encode=None,
        try_reload=False,
    )
    z_train = modulations["z_train"]
    # z_train_extra = modulations["z_train_extra"]
    # z_test = modulations["z_test"]

    z_mean = einops.rearrange(
        z_train, "b l t -> (b t) l").mean(0).reshape(1, code_dim_coral, 1)
    z_std = einops.rearrange(
        z_train, "b l t -> (b t) l").std(0).reshape(1, code_dim_coral, 1)

    # z_train = (z_train - z_mean) / z_std
    # z_train_extra = (z_train_extra - z_mean) / z_std
    # z_test = (z_test - z_mean) / z_std

    tmp_dyn = torch.load(root_dir / "model" / f"{dyn_run_name}.pt")
    cfg = tmp_dyn['cfg']
    dyn_state_dict = tmp_dyn['model']
    c = z_train.shape[2] if multichannel else 1
    dyn = Derivative_coral(
        c, z_train.shape[1], width_dyn_coral, depth_dyn_coral).cuda()
    dyn.load_state_dict(dyn_state_dict)

    print("dyn model loaded at epoch : ", tmp_dyn['epoch'])
    tmp_inr = torch.load(root_dir / "inr" / f"{inr_run_name}.pt")
    print("inr model loaded at epoch : ", tmp_inr['epoch'])

    if return_mod:
        return inr, alpha, dyn, z_mean, z_std, modulations

    return inr, alpha, dyn, z_mean, z_std


def load_dino(root_dir, dino_run_name):

    checkpoint = torch.load(root_dir / "dino" / f"{dino_run_name}_tr.pt")

    net_dec_params = checkpoint["net_dec_params"]
    state_dim = net_dec_params["state_c"]
    code_dim_dino = net_dec_params["code_c"]
    if 'model' not in net_dec_params.keys():
        net_dec_params['model'] = 'mfn'
    net_dec = Decoder(**net_dec_params)
    net_dec_dict = net_dec.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint["dec_state_dict"].items() if k in net_dec_dict
    }
    net_dec_dict.update(pretrained_dict)
    net_dec.load_state_dict(net_dec_dict)
    print(dict(net_dec.named_parameters()).keys())

    net_dyn_params = checkpoint["net_dyn_params"]
    net_dyn = Derivative_dino(**net_dyn_params)
    net_dyn_dict = net_dyn.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint["dyn_state_dict"].items() if k in net_dyn_dict
    }
    net_dyn_dict.update(pretrained_dict)
    net_dyn.load_state_dict(net_dyn_dict)
    print(dict(net_dyn.named_parameters()).keys())

    states_params = checkpoint["states_params"]
    net_dec = net_dec.cuda()
    net_dyn = net_dyn.cuda()

    print("model logged at epoch : ", checkpoint['epoch'])

    return net_dec, net_dyn, states_params, code_dim_dino
