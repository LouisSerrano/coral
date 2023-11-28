import torch
import torch.nn as nn
import time
import hydra
import wandb
import einops
from pathlib import Path
import os

from mppde.baseline_coral.models_gnn import MP_PDE_Solver
from mppde.baseline_coral.utils_coral import PDE_CORAL, GraphCreator
from mppde.baseline_coral.models_gnn_louis import MP_PDE_Solver_Louis


def load_mppde(root_dir, mppde_run_name, spatial_size, T, T_EXT, grid_type, nb_dim, pos_dim, input_dim, dt=1, batch_size=1):
    checkpoint = torch.load(root_dir / "mppde" / f"{mppde_run_name}.pt")
    cfg = checkpoint['cfg']

    print("model loaded at epoch : ", checkpoint['epoch'])

    pde = PDE_CORAL(pos_dim, input_dim, T, spatial_size,
                    0, T, T_EXT, dt, grid_type, batch_size)

    graph_creator = GraphCreator(pde=pde,
                                 neighbors=cfg.optim.neighbors,
                                 time_window=cfg.optim.time_window,
                                 x_resolution=spatial_size)

    mppde = MP_PDE_Solver(pde=pde,
                          hidden_features=cfg.model.hidden_features,
                          time_window=graph_creator.tw)
    mppde.load_state_dict(checkpoint['model'])

    return mppde, graph_creator, pde


def load_mppde_louis(root_dir, mppde_run_name, pos_dim, input_dim, output_dim):
    checkpoint = torch.load(root_dir / "mppde" / f"{mppde_run_name}.pt")
    cfg = checkpoint['cfg']

    print("model loaded at epoch : ", checkpoint['epoch'])

    mppde = MP_PDE_Solver_Louis(pos_dim,
                                input_dim,
                                output_dim,
                                cfg.optim.time_window,
                                cfg.model.hidden_features,
                                hidden_layer=6)
    mppde.load_state_dict(checkpoint['model'])

    return mppde
