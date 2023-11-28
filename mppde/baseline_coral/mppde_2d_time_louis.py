from mppde.baseline_coral.scheduling import mppde_pushforward, mppde_test_rollout
from mppde.baseline_coral.load_data import get_dynamics_data, set_seed
from mppde.baseline_coral.dynamics_dataset import GraphTemporalDataset
from mppde.baseline_coral.models_gnn_louis import MP_PDE_Solver_Louis
from mppde.baseline_coral.eval_mppde_louis import eval_mppde

from omegaconf import DictConfig, OmegaConf
import wandb
import torch
import numpy as np
import hydra
import einops
import os
from pathlib import Path
from torch import optim

from torch_geometric.loader import DataLoader


@hydra.main(config_path="config/", config_name="mppde.yaml")
def main(cfg: DictConfig) -> None:

    checkpoint_path = None if cfg.optim.checkpoint_path == "" else cfg.optim.checkpoint_path
    dataset_name = cfg.data.dataset_name
    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name
    mppde_run_name = checkpoint_path

    if checkpoint_path is not None:
        checkpoint = torch.load(root_dir / "mppde" / f"{mppde_run_name}.pt")
        cfg = checkpoint['cfg']

    # data
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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "mppde"

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    epochs = cfg.optim.epochs
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
    sweep_id = cfg.wandb.sweep_id

    if isinstance(sub_tr, float):
        grid_type = 'irregular'
    else:
        grid_type = 'regular'

    # model
    model_type = cfg.model.model_type
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
            dataset_name / data_to_encode / "mppde"
        )
    else:
        model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "mppde"

    os.makedirs(str(model_dir), exist_ok=True)

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

    run.tags = (
        ("mppde-louis",) +
        (dataset_name,) + (f"sub={sub_tr}",)
        + ("debug",)
    )

    if data_to_encode == None:
        run.tags = ("mppde-louis",) + (model_type,) + \
            (dataset_name,) + (f"sub={sub_tr}",) + ("debug",)
    else:
        run.tags = (
            ("mppde-louis",)
            + (model_type,)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
        )

    # flatten spatial dims
    u_train = einops.rearrange(u_train, 'B ... C T -> B (...) C T')
    grid_tr = einops.rearrange(grid_tr, 'B ... C T -> B (...) C T')  # * 0.5
    u_test = einops.rearrange(u_test, 'B ... C T -> B (...) C T')
    grid_te = einops.rearrange(grid_te, 'B ... C T -> B (...) C T')  # * 0.5
    if u_train_ext is not None:
        u_train_ext = einops.rearrange(u_train_ext, 'B ... C T -> B (...) C T')
        grid_tr_ext = einops.rearrange(
            grid_tr_ext, 'B ... C T -> B (...) C T')  # * 0.5
        u_test_ext = einops.rearrange(u_test_ext, 'B ... C T -> B (...) C T')
        grid_te_ext = einops.rearrange(
            grid_te_ext, 'B ... C T -> B (...) C T')  # * 0.5

    print("u_train.shape, grid_tr.shape : ", u_train.shape, grid_tr.shape)

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
    if u_train_ext is not None:
        trainset_ext = GraphTemporalDataset(
            u_train_ext, grid_tr_ext)
    if u_test_ext is not None:
        testset_ext = GraphTemporalDataset(
            u_test_ext, grid_te_ext)

    dt = 1
    timestamps = torch.arange(0, T, dt).float().cuda()  # 0.1

    # print("trainset[0][0].images.shape : ", trainset[0][0].images.shape)
    # print("trainset[0][0].pos_t.shape : ", trainset[0][0].pos_t.shape)

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
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
    )
    if u_train_ext is not None:
        train_loader_ext = DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )
    if u_test_ext is not None:
        test_loader_ext = DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )

    model = MP_PDE_Solver_Louis(pos_dim=2,
                                input_dim=1,
                                output_dim=1,
                                time_window=time_window,
                                hidden_features=hidden_features,
                                hidden_layer=6).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr)  # , weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[unrolling, 2000, 4000, 6000, 8000], gamma=lr_decay)

    radius = 0.25
    max_neighbours = neighbors

    best_loss = np.inf

    for step in range(epochs):
        pred_train_mse = 0
        code_train_mse = 0
        code_test_mse = 0
        step_show = step % 1 == 0

        for substep, (graph, idx) in enumerate(train_loader):
            model.train()
            n_samples = len(idx)

            graph = graph.cuda()

            loss = mppde_pushforward(model, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            if step_show:
                u_pred = mppde_test_rollout(model, graph, bundle_size=1)
                pred_train_mse += ((u_pred - graph.images)
                                   ** 2).mean() * n_samples

        scheduler.step()

        code_train_mse = code_train_mse / ntrain
        pred_train_mse = pred_train_mse / ntrain

        if step_show:
            pred_train_all_mse, pred_train_inter_mse, pred_train_extra_mse = eval_mppde(
                model, train_loader_ext, ntrain, sequence_length_in, sequence_length_out)
            pred_test_all_mse, pred_test_inter_mse, pred_test_extra_mse = eval_mppde(
                model, test_loader_ext, ntest, sequence_length_in, sequence_length_out)

            log_dic = {
                "pred_train_mse": pred_train_mse,
                "code_train_mse": code_train_mse,
                "pred_train_inter_mse": pred_train_inter_mse,
                "pred_train_all_mse": pred_train_all_mse,
                "pred_train_extra_mse": pred_train_extra_mse,
                "pred_test_inter_mse": pred_test_inter_mse,
                "pred_test_extra_mse": pred_test_extra_mse,
                "pred_test_all_mse": pred_test_all_mse,
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
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return code_test_mse


if __name__ == "__main__":
    main()
