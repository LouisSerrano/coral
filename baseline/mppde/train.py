from utils import mppde_create_data, mppde_pushforward, mppde_test_rollout, GraphTemporalDataset, KEY_TO_INDEX
from coral.utils.data.load_data import get_dynamics_data, set_seed
from model import MP_PDE_Solver
from omegaconf import DictConfig, OmegaConf
import wandb
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import hydra
import einops
import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo
from torch.nn.utils import weight_norm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import radius_graph

@hydra.main(config_path="", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # submitit.JobEnvironment()
    # data
    dataset_name = cfg.data.dataset_name
    data_dir = cfg.data.dir
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_tr = cfg.data.sub_tr
    sub_from = cfg.data.sub_from
    sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

    # optim
    batch_size = cfg.optim.batch_size
    print_interval=cfg.optim.print_interval
    lr = cfg.optim.lr
    neighbors = cfg.optim.neighbors
    time_window = cfg.optim.time_window
    unrolling = cfg.optim.unrolling
    lr_decay = cfg.optim.lr_decay
    epochs = cfg.optim.epochs

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name
    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )

    # model
    hidden_features = cfg.model.hidden_features

    print("run dir given", run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )

    if run_dir is not None:
        os.symlink(run.dir.split("/files")[0], run_dir)

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name

    if data_to_encode is not None:
        model_dir = (
            Path(os.getenv("WANDB_DIR")) /
            dataset_name / data_to_encode / "model"
        )
    else:
        model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "model"

    set_seed(seed)

    (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        seq_inter_len=seq_inter_len,
        seq_extra_len=seq_extra_len,
        sub_from=sub_from,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
    )
    
    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_te: {grid_te.shape}")

    run.tags = (
            ("mppde",) +
            (dataset_name,) + (f"sub={sub_tr}",)
        )

    # total frames = num_trajectories * sequence_length
    T = u_train.shape[-1]

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset = GraphTemporalDataset(
        u_train, grid_tr
    )
    testset = GraphTemporalDataset(
        u_test, grid_te
    )

    dt = 1
    timestamps = torch.arange(0, T, dt).float().cuda() #0.1
    T_in = 20
    T_out = 40

    # create torch dataset
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    if dataset_name == "navier-stokes-dino":
        pos_dim = 2
        input_dim = 1
        output_dim = 1
        time_window = 1

    elif dataset_name == "shallow-water-dino":
        pos_dim = 3
        input_dim = 2 
        output_dim = 2
        time_window = 1
    
    model = MP_PDE_Solver(pos_dim=pos_dim,
                          input_dim=input_dim,
                          output_dim=output_dim,
                          time_window=time_window,
                          hidden_features=hidden_features,
                          hidden_layer=6).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1000, 5000, 10000], gamma=0.4)
     
    best_loss = np.inf

    for step in range(epochs):
        pred_train_mse = 0
        pred_test_mse = 0
        pred_test_out_mse = 0
        pred_test_in_mse = 0
        pred_test_mse = 0
        code_train_mse = 0
        code_test_mse = 0
        step_show = step % 100 == 0

        for substep, (graph, idx) in enumerate(train_loader):
            model.train()
            n_samples = len(graph)

            graph.images = graph.images[..., :T_in]

            graph = graph.cuda()

            loss = mppde_pushforward(model, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples
            
            if step_show:
                u_pred = mppde_test_rollout(model, graph, bundle_size=1)
                pred_train_mse += ((u_pred - graph.images) ** 2).mean() * n_samples

        code_train_mse = code_train_mse / ntrain
        scheduler.step()

        if step_show:
            pred_train_mse = pred_train_mse / ntrain

        if step_show:
            for graph, idx in test_loader:
                model.eval()
                n_samples = len(graph)

                #print('test, graph.images', graph.images.shape, graph.batch.shape, graph.pos.shape)

                graph = graph.cuda()
                with torch.no_grad():
                    loss = mppde_pushforward(model, graph)
                code_test_mse += loss.item() * n_samples

                with torch.no_grad():
                    u_pred = mppde_test_rollout(model, graph, bundle_size=1)
                    pred_test_mse += ((u_pred - graph.images) ** 2).mean() * n_samples
                    pred_test_in_mse += ((u_pred[..., :T_in] - graph.images[..., :T_in]) ** 2).mean() * n_samples
                    pred_test_out_mse += ((u_pred[..., T_in:] - graph.images[..., T_in:]) ** 2).mean() * n_samples

            code_test_mse = code_test_mse / ntest
            pred_test_mse = pred_test_mse / ntest
            pred_test_in_mse = pred_test_in_mse / ntest
            pred_test_out_mse = pred_test_out_mse / ntest

        if step_show:
            log_dic = {
                "pred_test_mse": pred_test_mse,
                "pred_train_mse": pred_train_mse,
                "pred_test_int_mse": pred_test_in_mse,
                "pred_test_out_mse": pred_test_out_mse,
                "code_test_mse": code_test_mse,
                "code_train_mse": code_train_mse,
            }
        
            wandb.log(log_dic)

        else:
            wandb.log(
                {
                    "code_train_mse": code_train_mse,
                },
                step=step,
                commit=not step_show,
            )

        if code_train_mse < best_loss:
            best_loss = code_train_mse

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": code_test_mse,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{model_dir}/{run_name}.pt",
            )

    return code_test_mse


if __name__ == "__main__":
    main()