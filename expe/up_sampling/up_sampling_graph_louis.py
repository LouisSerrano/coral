import torch
import torch.nn as nn
import time
import hydra
import wandb
import einops
from pathlib import Path
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader
import json
import numpy as np

from mppde.baseline_coral.dynamics_dataset import GraphTemporalDataset
from mppde.baseline_coral.load_data import get_dynamics_data, set_seed
from expe.load_models.load_models_graph import load_mppde_louis
from expe.forwards.forwards_graph import forward_mppde_louis
from expe.config.run_names import RUN_NAMES


@hydra.main(config_path="config/", config_name="upsampling.yaml")
def main(cfg: DictConfig) -> None:
    cuda = torch.cuda.is_available()
    if cuda:
        gpu_id = torch.cuda.current_device()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print("device : ", device)

    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name
    times_dir = '/home/lise.leboudec/project/coral/expe/config/'
    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    mppde_run_name = RUN_NAMES[sub_from][sub_tr]["mppde"]

    print("mppde_run_name : ", mppde_run_name)
    n_baselines = int(mppde_run_name != None)

    print(f"running on {n_baselines} baselines")

    if mppde_run_name is not None:
        cfg_mppde = torch.load(root_dir / "mppde" /
                               f"{mppde_run_name}.pt")['cfg']

    upsamplings = [sub_tr, '1', '2', '4']
    nr_gt_steps = 1

    # data
    ntrain = cfg_mppde.data.ntrain
    ntest = cfg_mppde.data.ntest
    data_to_encode = cfg_mppde.data.data_to_encode
    # sub_from = cfg_mppde.data.sub_from
    # sub_tr = cfg_mppde.data.sub_tr
    # sub_te = cfg_mppde.data.sub_te
    seed = cfg_mppde.data.seed
    same_grid = cfg_mppde.data.same_grid
    setting = cfg_mppde.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg_mppde.data.sequence_length_in
    sequence_length_out = cfg_mppde.data.sequence_length_out

    assert cfg.data.sub_from == cfg_mppde.data.sub_from, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_from} but model trained on {cfg_mppde.data.sub_from}"
    assert cfg.data.sub_tr == cfg_mppde.data.sub_tr, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_tr} but model trained on {cfg_mppde.data.sub_tr}"
    assert cfg.data.sub_te == cfg_mppde.data.sub_te, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_te} but model trained on {cfg_mppde.data.sub_te}"

    print(
        f"running from setting {setting} with sampling {sub_from} / {sub_tr} - {sub_te}")
    if isinstance(sub_tr, float):
        grid_type = 'irregular'
    else:
        grid_type = 'regular'

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    criterion = nn.MSELoss()

    if dataset_name == 'shallow-water-dino':
        multichannel = True
    else:
        multichannel = False

    # experiments
    sub_from1 = 4
    sub_from2 = 2
    sub_from3 = 1

    set_seed(seed)

    (u_train, u_test, grid_tr, grid_te, _, _, _, _, u_train_ext, u_test_ext, grid_tr_ext, grid_te_ext) = get_dynamics_data(
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

    (_, _, _, _, _, _, _, _, u_train_up1, u_test_up1, grid_tr_up1, grid_te_up1) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length_optim,
        sub_from=sub_from1,
        sub_tr=1,
        sub_te=1,
        same_grid=same_grid,
        setting=setting,
        sequence_length_in=sequence_length_in,
        sequence_length_out=sequence_length_out
    )

    (_, _, _, _, _, _, _, _, u_train_up4, u_test_up4, grid_tr_up4, grid_te_up4) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length_optim,
        sub_from=sub_from2,
        sub_tr=1,
        sub_te=1,
        same_grid=same_grid,
        setting=setting,
        sequence_length_in=sequence_length_in,
        sequence_length_out=sequence_length_out
    )

    (_, _, _, _, _, _, _, _, u_train_up16, u_test_up16, grid_tr_up16, grid_te_up16) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length_optim,
        sub_from=sub_from3,
        sub_tr=1,
        sub_te=1,
        same_grid=same_grid,
        setting=setting,
        sequence_length_in=sequence_length_in,
        sequence_length_out=sequence_length_out
    )

    # flatten spatial dims
    u_train = einops.rearrange(u_train, 'B ... C T -> B (...) C T')
    grid_tr = einops.rearrange(grid_tr, 'B ... C T -> B (...) C T')
    u_test = einops.rearrange(u_test, 'B ... C T -> B (...) C T')
    grid_te = einops.rearrange(grid_te, 'B ... C T -> B (...) C T')
    if u_train_ext is not None:
        u_train_ext = einops.rearrange(u_train_ext, 'B ... C T -> B (...) C T')
        grid_tr_ext = einops.rearrange(
            grid_tr_ext, 'B ... C T -> B (...) C T')
        u_test_ext = einops.rearrange(u_test_ext, 'B ... C T -> B (...) C T')
        grid_te_ext = einops.rearrange(
            grid_te_ext, 'B ... C T -> B (...) C T')
    if u_train_up1 is not None:
        u_train_up1 = einops.rearrange(u_train_up1, 'B ... C T -> B (...) C T')
        grid_tr_up1 = einops.rearrange(
            grid_tr_up1, 'B ... C T -> B (...) C T')
        u_test_up1 = einops.rearrange(u_test_up1, 'B ... C T -> B (...) C T')
        grid_te_up1 = einops.rearrange(
            grid_te_up1, 'B ... C T -> B (...) C T')
    if u_train_up4 is not None:
        u_train_up4 = einops.rearrange(u_train_up4, 'B ... C T -> B (...) C T')
        grid_tr_up4 = einops.rearrange(
            grid_tr_up4, 'B ... C T -> B (...) C T')
        u_test_up4 = einops.rearrange(u_test_up4, 'B ... C T -> B (...) C T')
        grid_te_up4 = einops.rearrange(
            grid_te_up4, 'B ... C T -> B (...) C T')
    if u_train_up16 is not None:
        u_train_up16 = einops.rearrange(
            u_train_up16, 'B ... C T -> B (...) C T')
        grid_tr_up16 = einops.rearrange(
            grid_tr_up16, 'B ... C T -> B (...) C T')
        u_test_up16 = einops.rearrange(u_test_up16, 'B ... C T -> B (...) C T')
        grid_te_up16 = einops.rearrange(
            grid_te_up16, 'B ... C T -> B (...) C T')

    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_te: {grid_te.shape}")
    if u_train_ext is not None:
        print(
            f"data: {dataset_name}, u_train_ext: {u_train_ext.shape}, u_test_ext: {u_test_ext.shape}")
        print(
            f"grid: grid_tr_ext: {grid_tr_ext.shape}, grid_te_ext: {grid_te_ext.shape}")
    if u_train_up1 is not None:
        print(
            f"data: {dataset_name}, u_train_up1: {u_train_up1.shape}, u_test_up1: {u_test_up1.shape}")
        print(
            f"grid: grid_tr_up1: {grid_tr_up1.shape}, grid_te_up1: {grid_te_up1.shape}")
    if u_train_up4 is not None:
        print(
            f"data: {dataset_name}, u_train_up4: {u_train_up4.shape}, u_test_up4: {u_test_up4.shape}")
        print(
            f"grid: grid_tr_up4: {grid_tr_up4.shape}, grid_te_up4: {grid_te_up4.shape}")
    if u_train_up16 is not None:
        print(
            f"data: {dataset_name}, u_train_up16: {u_train_up16.shape}, u_test_up16: {u_test_up16.shape}")
        print(
            f"grid: grid_tr_up16: {grid_tr_up16.shape}, grid_te_up16: {grid_te_up16.shape}")

    n_seq_train = u_train.shape[0]
    n_seq_test = u_test.shape[0]
    spatial_size = u_train.shape[2]  # 64 en dur
    state_dim = u_train.shape[3]  # N, XY, C, T
    coord_dim = grid_tr.shape[3]  # N, XY, C, T
    T = u_train.shape[1]

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset = GraphTemporalDataset(
        u_train, grid_tr)
    testset = GraphTemporalDataset(
        u_test, grid_te)
    if u_train_ext is not None:
        trainset_ext = GraphTemporalDataset(
            u_train_ext, grid_tr_ext)
    if u_test_ext is not None:
        testset_ext = GraphTemporalDataset(
            u_test_ext, grid_te_ext)
    if u_train_up1 is not None:
        trainset_up1 = GraphTemporalDataset(
            u_train_up1, grid_tr_up1)
    if u_test_up1 is not None:
        testset_up1 = GraphTemporalDataset(
            u_test_up1, grid_te_up1)
    if u_train_up4 is not None:
        trainset_up4 = GraphTemporalDataset(
            u_train_up4, grid_tr_up4)
    if u_test_up4 is not None:
        testset_up4 = GraphTemporalDataset(
            u_test_up4, grid_te_up4)
    if u_train_up16 is not None:
        trainset_up16 = GraphTemporalDataset(
            u_train_up16, grid_tr_up16)
    if u_test_up16 is not None:
        testset_up16 = GraphTemporalDataset(
            u_test_up16, grid_te_up16)

    # create torch dataset
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    if u_train_ext is not None:
        train_loader_ext = DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_ext is not None:
        test_loader_ext = DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up1 is not None:
        train_loader_up1 = DataLoader(
            trainset_up1,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up1 is not None:
        test_loader_up1 = DataLoader(
            testset_up1,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up4 is not None:
        train_loader_up4 = DataLoader(
            trainset_up4,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up4 is not None:
        test_loader_up4 = DataLoader(
            testset_up4,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up16 is not None:
        train_loader_up16 = DataLoader(
            trainset_up16,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up16 is not None:
        test_loader_up16 = DataLoader(
            testset_up16,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

    T = u_train.shape[1]
    if u_test_ext is not None:
        T_EXT = u_test_ext.shape[1]

    dt = 1
    timestamps_train = torch.arange(0, T, dt).float().cuda()
    timestamps_ext = torch.arange(0, T_EXT, dt).float().cuda()

    input_dim = 1
    output_dim = 1
    pos_dim = 2

    mppde = load_mppde_louis(root_dir, mppde_run_name,
                             pos_dim, input_dim, output_dim)
    mppde = mppde.cuda()

    losses_train = torch.zeros((n_baselines, 4, 2)).cuda()
    times_train_mppde = torch.zeros((4)).cuda()
    losses_test = torch.zeros((n_baselines, 4, 2)).cuda()
    times_test_mppde = torch.zeros((4)).cuda()
    torch.set_printoptions(precision=10)
    with open(times_dir + 'times.json', 'r') as f:
        TIMES = json.load(f)
    with open(times_dir + 'times_up.json', 'r') as f:
        TIMES_UP = json.load(f)

    print('--- Evaluation on train dataset ---')
    for up, loader in enumerate([train_loader_ext, train_loader_up1, train_loader_up4, train_loader_up16]):
        print("up : ", upsamplings[up])
        for bidx, batch in enumerate(loader):
            images = batch[0].images.cuda()
            n_samples = len(batch[1])
            i = 0
            if mppde_run_name is not None:
                tic = time.time()
                pred_coral = forward_mppde_louis(
                    mppde, batch)
                tac = time.time()
                losses_train[i, up, 0] += ((pred_coral[..., :sequence_length_in] -
                                            images[..., :sequence_length_in]) ** 2).mean() * n_samples
                losses_train[i, up, 1] += ((pred_coral[..., sequence_length_in:sequence_length_in+sequence_length_out] -
                                            images[..., sequence_length_in:sequence_length_in+sequence_length_out]) ** 2).mean() * n_samples
                times_train_mppde[up] += (tac - tic)
                i += 1
        i = 0
        if mppde_run_name is not None:
            print(
                f"MPPDE train up sampling rate {upsamplings[up]} in-t: {losses_train[i, up, 0] / n_seq_train}")
            print(
                f"MPPDE train up sampling rate {upsamplings[up]} out-t: {losses_train[i, up, 1] / n_seq_train}")
            print(
                f"MPPDE train mean inference time sampling rate {upsamplings[up]} for batch_size {batch_size} : {times_train_mppde[up] / n_seq_train}")
            i += 1

    losses_train = losses_train.cpu() / n_seq_train
    times_train_mppde = times_train_mppde.cpu().numpy().astype(np.float64) / n_seq_train
    print(losses_train)

    if sub_tr == 0.05:
        TIMES_UP["mppde"]["train"][f'{upsamplings[0]}'] = times_train_mppde[0]
        TIMES_UP["mppde"]["train"][f'{upsamplings[1]}'] = times_train_mppde[1]
        TIMES_UP["mppde"]["train"][f'{upsamplings[2]}'] = times_train_mppde[2]
        TIMES_UP["mppde"]["train"][f'{upsamplings[3]}'] = times_train_mppde[3]
    TIMES["mppde"]["train"][f'{sub_tr}'] = times_train_mppde[0]

    print('--- Evaluation on test dataset ---')
    for up, loader in enumerate([test_loader_ext, test_loader_up1, test_loader_up4, test_loader_up16]):
        print("up : ", upsamplings[up])
        for bidx, batch in enumerate(loader):
            images = batch[0].images.cuda()
            n_samples = len(batch[1])
            i = 0
            if mppde_run_name is not None:
                tic = time.time()
                pred_coral = forward_mppde_louis(
                    mppde, batch)
                tac = time.time()
                losses_test[i, up, 0] += ((pred_coral[..., :sequence_length_in] -
                                           images[..., :sequence_length_in]) ** 2).mean() * n_samples
                losses_test[i, up, 1] += ((pred_coral[..., sequence_length_in:sequence_length_in+sequence_length_out] -
                                           images[..., sequence_length_in:sequence_length_in+sequence_length_out]) ** 2).mean() * n_samples
                times_test_mppde[up] += (tac - tic)
                i += 1
        i = 0
        if mppde_run_name is not None:
            print(
                f"MPPDE test up sampling rate {upsamplings[up]} in-t: {losses_test[i, up, 0] / n_seq_test}")
            print(
                f"MPPDE test up sampling rate {upsamplings[up]} out-t: {losses_test[i, up, 1] / n_seq_test}")
            print(
                f"MPPDE test mean inference time sampling rate {upsamplings[up]} for batch_size {batch_size} : {times_test_mppde[up] / n_seq_test}")
            i += 1
    losses_test = losses_test.cpu() / n_seq_test
    times_test_mppde = times_test_mppde.cpu().numpy().astype(np.float64) / n_seq_test
    print(losses_test)

    if sub_tr == 0.05:
        TIMES_UP["mppde"]["test"][f'{upsamplings[0]}'] = times_test_mppde[0]
        TIMES_UP["mppde"]["test"][f'{upsamplings[1]}'] = times_test_mppde[1]
        TIMES_UP["mppde"]["test"][f'{upsamplings[2]}'] = times_test_mppde[2]
        TIMES_UP["mppde"]["test"][f'{upsamplings[3]}'] = times_test_mppde[3]
    TIMES["mppde"]["test"][f'{sub_tr}'] = times_test_mppde[0]

    with open(times_dir + 'times.json', 'w') as f:
        json.dump(TIMES, f, indent=4)
    with open(times_dir + 'times_up.json', 'w') as f:
        json.dump(TIMES_UP, f, indent=4)


if __name__ == "__main__":
    main()
