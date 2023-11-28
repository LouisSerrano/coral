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

from mppde.baseline_coral.dynamics_dataset import GraphTemporalDataset
from mppde.baseline_coral.load_data import get_dynamics_data, set_seed
from expe.load_models.load_models_graph import load_mppde_louis
from expe.forwards.forwards_graph import forward_mppde_louis
from expe.config.run_names import RUN_NAMES
from expe.visualization.visualization_functions import plot_baselines, gif_baselines, plot_errors, plot_grid, save_imshow, save_scatter


@hydra.main(config_path="config/", config_name="visualization.yaml")
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
        baselines = ['mppde']

    upsamplings = ['0', '1', '2', '4']
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
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    if u_train_ext is not None:
        train_loader_ext = DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_ext is not None:
        test_loader_ext = DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up1 is not None:
        train_loader_up1 = DataLoader(
            trainset_up1,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up1 is not None:
        test_loader_up1 = DataLoader(
            testset_up1,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up4 is not None:
        train_loader_up4 = DataLoader(
            trainset_up4,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up4 is not None:
        test_loader_up4 = DataLoader(
            testset_up4,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up16 is not None:
        train_loader_up16 = DataLoader(
            trainset_up16,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up16 is not None:
        test_loader_up16 = DataLoader(
            testset_up16,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )

    T = u_train.shape[1]
    if u_test_ext is not None:
        T_EXT = u_test_ext.shape[1]

    dt = 1

    input_dim = 1
    output_dim = 1
    pos_dim = 2

    mppde = load_mppde_louis(root_dir, mppde_run_name,
                             pos_dim, input_dim, output_dim)
    mppde = mppde.cuda()

    torch.set_printoptions(precision=10)

    # b0 = next(iter(test_loader_ext))
    batch = next(iter(test_loader_ext))
    idx = batch[1]
    b0 = batch[0].images.unsqueeze(0)
    x = grid_te_ext[0, ..., 0, 0]  # XY
    y = grid_te_ext[0, ..., 1, 0]  # XY
    pred_mppde0 = forward_mppde_louis(
        mppde, batch).unsqueeze(0).cpu().detach().numpy()
    # b1 = next(iter(test_loader_up1))
    batch = next(iter(test_loader_up1))
    b1 = batch[0].images.unsqueeze(0)
    x1 = grid_te_up1[0, ..., 0, 0]  # XY
    y1 = grid_te_up1[0, ..., 1, 0]  # XY
    assert batch[1] == idx
    pred_mppde1 = forward_mppde_louis(
        mppde, batch).unsqueeze(0).cpu().detach().numpy()
    # b4 = next(iter(test_loader_up4))
    batch = next(iter(test_loader_up4))
    b4 = batch[0].images.unsqueeze(0)
    assert batch[1] == idx
    pred_mppde4 = forward_mppde_louis(
        mppde, batch).unsqueeze(0).cpu().detach().numpy()
    batch = next(iter(test_loader_up16))
    b16 = batch[0].images.unsqueeze(0)
    assert batch[1] == idx
    # b16 = next(iter(test_loader_up16))
    pred_mppde16 = forward_mppde_louis(
        mppde, batch).unsqueeze(0).cpu().detach().numpy()

    time2show = 30
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    print("b0.shape : ", b0.shape)
    print("b1.shape : ", b1.shape)
    print("b4.shape : ", b4.shape)
    print("b16.shape : ", b16.shape)
    print("pred_mppde0.shape : ", pred_mppde0.shape)  # B, XY, C, T
    print("pred_mppde1.shape : ", pred_mppde1.shape)
    print("pred_mppde4.shape : ", pred_mppde4.shape)
    print("pred_mppde16.shape : ", pred_mppde16.shape)

    pred0 = (pred_mppde0, )
    pred1 = (pred_mppde1, )
    pred4 = (pred_mppde4, )
    pred16 = (pred_mppde16, )

    title = f'ns-upsampling-{sub_tr}-64to256_graph.png'
    plot_baselines(plot_dir, title, b0, b1, b4, b16, pred0,
                   pred1, pred4, pred16, x, y, time2show, baselines)

    title = f'ns-upsampling-{sub_tr}-64to256_true_graph.gif'
    gif_baselines(plot_dir, title, b0, b1, b4, b16, x, y)
    title = f'ns-upsampling-{sub_tr}-64to256_mppde.gif'
    gif_baselines(plot_dir, title, pred_mppde0, pred_mppde1,
                  pred_mppde4, pred_mppde16, x, y)

    title = f'ns-errors-{sub_tr}-64to256_graph.png'
    plot_errors(plot_dir, title, b0, b1, b4, b16, pred0,
                pred1, pred4, pred16, x, y, baselines)

    title = f'ns-grid-{sub_tr}.png'
    plot_grid(plot_dir + 'mppde/', title, x, y, x1, y1)

    times2show = range(0, 40)
    title = f'ns-true-{sub_tr}'
    save_imshow(plot_dir + 'true/', title, b1, times2show, 64)

    title = f'ns-mppde-{sub_tr}'
    save_imshow(plot_dir + 'mppde/', title, pred1[0], times2show, 64)


if __name__ == "__main__":
    main()
