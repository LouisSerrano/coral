import torch
import os
import sys
import einops
import wandb
import hydra

import numpy as np

from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from pathlib import Path

from mppde.baseline_coral.utils_coral import GraphCreator, TemporalDatasetWithCode, KEY_TO_INDEX, PDE_CORAL
from mppde.baseline_coral.models_gnn import MP_PDE_Solver
from mppde.baseline_coral.load_data import get_dynamics_data, set_seed
from mppde.baseline_coral.train import train, test


@hydra.main(config_path="config/", config_name="mppde.yaml")
def main(cfg: DictConfig) -> None:

    checkpoint_path = None if cfg.optim.checkpoint_path == "" else cfg.optim.checkpoint_path
    dataset_name = cfg.data.dataset_name
    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name
    mppde_run_name = checkpoint_path

    if checkpoint_path is not None:
        checkpoint = torch.load(root_dir / "mppde" / f"{mppde_run_name}.pt")
        cfg = checkpoint['cfg']

    data_dir = cfg.data.dir
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    setting = cfg.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg.data.sequence_length_in
    sequence_length_out = cfg.data.sequence_length_out

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    print_interval = cfg.optim.print_interval
    lr = cfg.optim.learning_rate
    neighbors = cfg.optim.neighbors
    time_window = cfg.optim.time_window
    unrolling = cfg.optim.unrolling
    lr_decay = cfg.optim.lr_decay
    weight_decay = cfg.optim.weight_decay
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

    if isinstance(sub_tr, float):
        grid_type = 'irregular'
    else:
        grid_type = 'regular'

    # model
    model_type = cfg.model.model_type
    hidden_features = cfg.model.hidden_features

    # cuda
    device = "cuda"

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=run_dir,
    )

    run_name = wandb.run.name

    print("run dir given", run_dir)
    print("id", run.id)
    print("dir", run.dir)

    if run_dir is not None:
        os.symlink(run.dir.split("/files")[0], run_dir)

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

    if data_to_encode is not None:
        model_dir = (
            Path(os.getenv("WANDB_DIR")) /
            dataset_name / data_to_encode / "mppde"
        )
    else:
        model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "mppde"

    os.makedirs(str(model_dir), exist_ok=True)

    # set seed
    set_seed(seed)

    (u_train, u_test, grid_tr, grid_te, u_train_out, u_test_out, grid_tr_out, grid_te_out, u_train_ext, u_test_ext, grid_tr_ext, grid_te_ext) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length_optim,
        sub_from=sub_from,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
        setting=setting,
        sequence_length_in=sequence_length_in,
        sequence_length_out=sequence_length_out
    )

    # for evaluation in setting all
    if u_train_ext is None:
        u_train_ext = u_train
        grid_tr_ext = grid_tr
    if u_test_ext is None:
        u_test_ext = u_test
        grid_te_ext = grid_te

    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_te: {grid_te.shape}")
    if u_train_out is not None:
        print(
            f"data: {dataset_name}, u_train_out: {u_train_out.shape}, u_test_out: {u_test_out.shape}")
        print(
            f"grid: grid_tr_out: {grid_tr_out.shape}, grid_te_out: {grid_te_out.shape}")

    if data_to_encode == None:
        run.tags = ("mppde",) + (model_type,) + \
            (dataset_name,) + (f"sub={sub_tr}",) + ("debug",)
    else:
        run.tags = (
            ("mppde",)
            + (model_type,)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
            + ("debug",)
        )

    # flatten spatial dims
    u_train = einops.rearrange(u_train, 'B ... C T -> B T (...) C')
    grid_tr = einops.rearrange(grid_tr, 'B ... C T -> B T (...) C')  # * 0.5
    u_test = einops.rearrange(u_test, 'B ... C T -> B T (...) C')
    grid_te = einops.rearrange(grid_te, 'B ... C T -> B T (...) C')  # * 0.5
    if u_train_ext is not None:
        u_train_ext = einops.rearrange(u_train_ext, 'B ... C T -> B T (...) C')
        grid_tr_ext = einops.rearrange(
            grid_tr_ext, 'B ... C T -> B T (...) C')  # * 0.5
        u_test_ext = einops.rearrange(u_test_ext, 'B ... C T -> B T (...) C')
        grid_te_ext = einops.rearrange(
            grid_te_ext, 'B ... C T -> B T (...) C')  # * 0.5

    print("u_train.shape, grid_tr.shape : ", u_train.shape, grid_tr.shape)

    spatial_size = u_train.shape[2]  # 64*64
    T = u_train.shape[1]
    T_EXT = u_train_ext.shape[1]
    nb_dim = grid_tr.shape[-1]
    dt = 1

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, 0, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, 0, dataset_name, data_to_encode
    )
    if u_train_ext is not None:
        trainset_ext = TemporalDatasetWithCode(
            u_train_ext, grid_tr_ext, 0, dataset_name, data_to_encode)
    if u_test_ext is not None:
        testset_ext = TemporalDatasetWithCode(
            u_test_ext, grid_te_ext, 0, dataset_name, data_to_encode)
    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
    )
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )
    if u_test_ext is not None:
        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )

    input_dim = trainset[0][0].shape[-1]
    pos_dim = grid_tr.shape[-1]
    pde = PDE_CORAL(pos_dim, input_dim, T, spatial_size,
                    0, T, T_EXT, dt, grid_type, batch_size)

    # Equation specific input variables
    eq_variables = {}
    print("T : ", T)
    graph_creator = GraphCreator(pde=pde,
                                 neighbors=neighbors,
                                 time_window=time_window,
                                 x_resolution=spatial_size).to(device)
    if model_type == 'GNN':
        model = MP_PDE_Solver(pde=pde,
                              hidden_features=hidden_features,
                              time_window=graph_creator.tw).to(device)
    else:
        raise Exception("Wrong model specified")

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Number of parameters: {params}')

    # Optimizer
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[unrolling, 5, 10, 15], gamma=lr_decay)

    # Training loop
    best_loss = 10e30
    pred_train_inter_mse = 10e30
    criterion = torch.nn.MSELoss()

    epoch_range = range(epochs)

    if checkpoint_path is not None:
        checkpoint = torch.load(root_dir / "mppde" / f"{mppde_run_name}.pt")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        best_loss = checkpoint['loss_inter']
        epoch_range = range(epoch, epochs)

    for epoch in epoch_range:
        step_show = epoch % 100
        print(f"Epoch {epoch}")
        pred_train_mse = train(pde, epoch, model, optimizer, train_loader,
                               graph_creator, criterion, device=device, unrolling=unrolling,
                               ntrain=ntrain, print_interval=print_interval)
        scheduler.step()

        log_dic = { "pred_train_mse": pred_train_mse,
                    "code_train_mse": pred_train_mse,}

        if step_show:
            pred_train_inter_mse, pred_train_extra_mse = test(
                pde, model, train_loader_ext, graph_creator, criterion,  ntest=ntrain, device=device)
            pred_test_inter_mse, pred_test_extra_mse = test(
                pde, model, test_loader_ext, graph_creator, criterion, ntest=ntest, device=device)
            
            log_dic.update({"pred_train_inter_mse": pred_train_inter_mse,
                            "pred_train_extra_mse": pred_train_extra_mse,
                            "pred_test_inter_mse": pred_test_inter_mse,
                            "pred_test_extra_mse": pred_test_extra_mse,})

        wandb.log(log_dic)

        if pred_train_inter_mse < best_loss:
            best_loss = pred_train_inter_mse
            torch.save(
                {
                    "cfg": cfg,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss_inter": best_loss,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{model_dir}/{run_name}.pt",
            )
    return None


if __name__ == "__main__":
    main()
