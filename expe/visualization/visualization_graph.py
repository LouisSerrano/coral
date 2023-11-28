import torch
import torch.nn as nn
import einops
from pathlib import Path
import os
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
import omegaconf
import hydra

from mppde.baseline_coral.load_data import get_dynamics_data, set_seed
from expe.load_models.load_models_graph import load_mppde
from expe.forwards.forwards_graph import forward_mppde
from expe.config.run_names import RUN_NAMES
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode
from expe.visualization.visualization_functions import plot_baselines, gif_baselines


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

    assert mppde_run_name is not None, 'No run found'

    if mppde_run_name is not None:
        baselines = ['MPPDE']
        cfg_mppde = torch.load(root_dir / "mppde" /
                               f"{mppde_run_name}.pt")['cfg']

    upsamplings = ['0', '1', '2', '4']
    nr_gt_steps = 1

    # data
    ntrain = cfg_mppde.data.ntrain
    ntest = cfg_mppde.data.ntest
    data_to_encode = cfg_mppde.data.data_to_encode
    sub_tr = cfg_mppde.data.sub_tr
    sub_te = cfg_mppde.data.sub_te if sub_te is not None else sub_tr
    try:
        sub_from = cfg_mppde.data.sub_from
    except omegaconf.errors.ConfigAttributeError:
        sub_from = sub_tr  # firsts runs don't have a sub_from attribute ie run 4 / 1-1
        sub_tr = 1
        sub_te = 1
    seed = cfg_mppde.data.seed
    same_grid = cfg_mppde.data.same_grid
    setting = cfg_mppde.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg_mppde.data.sequence_length_in
    sequence_length_out = cfg_mppde.data.sequence_length_out

    if isinstance(sub_tr, float):
        grid_type = 'irregular'
    else:
        grid_type = 'regular'

    assert cfg.data.sub_from == cfg_mppde.data.sub_from, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_from} but model trained on {cfg_mppde.data.sub_from}"
    assert cfg.data.sub_tr == cfg_mppde.data.sub_tr, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_tr} but model trained on {cfg_mppde.data.sub_tr}"
    assert cfg.data.sub_te == cfg_mppde.data.sub_te, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_te} but model trained on {cfg_mppde.data.sub_te}"

    print(
        f"running from setting {setting} with sampling {sub_from} / {sub_tr} - {sub_te}")

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
    u_train = einops.rearrange(u_train, 'B ... C T -> B T (...) C')
    grid_tr = einops.rearrange(grid_tr, 'B ... C T -> B T (...) C')
    u_test = einops.rearrange(u_test, 'B ... C T -> B T (...) C')
    grid_te = einops.rearrange(grid_te, 'B ... C T -> B T (...) C')
    if u_train_ext is not None:
        u_train_ext = einops.rearrange(u_train_ext, 'B ... C T -> B T (...) C')
        grid_tr_ext = einops.rearrange(
            grid_tr_ext, 'B ... C T -> B T (...) C')
        u_test_ext = einops.rearrange(u_test_ext, 'B ... C T -> B T (...) C')
        grid_te_ext = einops.rearrange(
            grid_te_ext, 'B ... C T -> B T (...) C')
    if u_train_up1 is not None:
        u_train_up1 = einops.rearrange(u_train_up1, 'B ... C T -> B T (...) C')
        grid_tr_up1 = einops.rearrange(
            grid_tr_up1, 'B ... C T -> B T (...) C')
        u_test_up1 = einops.rearrange(u_test_up1, 'B ... C T -> B T (...) C')
        grid_te_up1 = einops.rearrange(
            grid_te_up1, 'B ... C T -> B T (...) C')
    if u_train_up4 is not None:
        u_train_up4 = einops.rearrange(u_train_up4, 'B ... C T -> B T (...) C')
        grid_tr_up4 = einops.rearrange(
            grid_tr_up4, 'B ... C T -> B T (...) C')
        u_test_up4 = einops.rearrange(u_test_up4, 'B ... C T -> B T (...) C')
        grid_te_up4 = einops.rearrange(
            grid_te_up4, 'B ... C T -> B T (...) C')
    if u_train_up16 is not None:
        u_train_up16 = einops.rearrange(
            u_train_up16, 'B ... C T -> B T (...) C')
        grid_tr_up16 = einops.rearrange(
            grid_tr_up16, 'B ... C T -> B T (...) C')
        u_test_up16 = einops.rearrange(u_test_up16, 'B ... C T -> B T (...) C')
        grid_te_up16 = einops.rearrange(
            grid_te_up16, 'B ... C T -> B T (...) C')

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

    n_seq_train = u_train.shape[0]  # 512 en dur
    n_seq_test = u_test.shape[0]  # 512 en dur
    spatial_size = u_train.shape[1]  # 64 en dur
    state_dim = u_train.shape[2]  # N, XY, C, T
    coord_dim = grid_tr.shape[2]  # N, XY, C, T
    T = u_train.shape[-1]

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
    if u_train_up1 is not None:
        trainset_up1 = TemporalDatasetWithCode(
            u_train_up1, grid_tr_up1, 0, dataset_name, data_to_encode)
    if u_test_up1 is not None:
        testset_up1 = TemporalDatasetWithCode(
            u_test_up1, grid_te_up1, 0, dataset_name, data_to_encode)
    if u_train_up4 is not None:
        trainset_up4 = TemporalDatasetWithCode(
            u_train_up4, grid_tr_up4, 0, dataset_name, data_to_encode)
    if u_test_up4 is not None:
        testset_up4 = TemporalDatasetWithCode(
            u_test_up4, grid_te_up4, 0, dataset_name, data_to_encode)
    if u_train_up16 is not None:
        trainset_up16 = TemporalDatasetWithCode(
            u_train_up16, grid_tr_up16, 0, dataset_name, data_to_encode)
    if u_test_up16 is not None:
        testset_up16 = TemporalDatasetWithCode(
            u_test_up16, grid_te_up16, 0, dataset_name, data_to_encode)

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_ext is not None:
        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up1 is not None:
        train_loader_up1 = torch.utils.data.DataLoader(
            trainset_up1,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up1 is not None:
        test_loader_up1 = torch.utils.data.DataLoader(
            testset_up1,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up4 is not None:
        train_loader_up4 = torch.utils.data.DataLoader(
            trainset_up4,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up4 is not None:
        test_loader_up4 = torch.utils.data.DataLoader(
            testset_up4,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up16 is not None:
        train_loader_up16 = torch.utils.data.DataLoader(
            trainset_up16,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up16 is not None:
        test_loader_up16 = torch.utils.data.DataLoader(
            testset_up16,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
            drop_last=True,
        )

    T = u_train.shape[-1]
    if u_test_ext is not None:
        T_EXT = u_test_ext.shape[-1]

    # trainset coords of shape (N, Dx, Dy, input_dim, T)
    input_dim = grid_tr.shape[-2]
    # trainset images of shape (N, Dx, Dy, output_dim, T)
    output_dim = u_train.shape[-2]

    dt = 1
    timestamps_train = torch.arange(0, T, dt).float().cuda()
    timestamps_ext = torch.arange(0, T_EXT, dt).float().cuda()

    input_dim = trainset[0][0].shape[-1]
    pos_dim = grid_tr.shape[-1]

    mppde, graph_creator, pde = load_mppde(
        root_dir, mppde_run_name, spatial_size, T, T_EXT, grid_type, coord_dim, pos_dim, input_dim, dt=1, batch_size=1)
    mppde = mppde.cuda()
    graph_creator = graph_creator.cuda()

    losses_train = torch.zeros((n_baselines, 4, 2)).cuda()
    losses_test = torch.zeros((n_baselines, 4, 2)).cuda()
    torch.set_printoptions(precision=10)

    idx = 0

    batch = next(iter(test_loader_ext))
    pred_mppde0 = forward_mppde(
        mppde, batch, graph_creator, nr_gt_steps, device).cpu().detach().numpy()
    batch = next(iter(test_loader_up1))
    pred_mppde1 = forward_mppde(
        mppde, batch, graph_creator, nr_gt_steps, device).cpu().detach().numpy()
    batch = next(iter(test_loader_up4))
    pred_mppde4 = forward_mppde(
        mppde, batch, graph_creator, nr_gt_steps, device).cpu().detach().numpy()
    batch = next(iter(test_loader_up16))
    pred_mppde16 = forward_mppde(
        mppde, batch, graph_creator, nr_gt_steps, device).cpu().detach().numpy()

    idx = [0]  # TODOOOOO idx = [idx] plutot
    time2show = 20
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    b0 = trainset_ext[idx]
    b1 = trainset_up1[idx]
    b4 = trainset_up4[idx]
    b16 = trainset_up16[idx]

    x = b0[2][0, ..., 0, time2show]  # XY
    y = b0[2][0, ..., 1, time2show]  # XY

    print("pred_mppde0.shape : ", pred_mppde0.shape)  # B, XY, C, T
    print("pred_mppde1.shape : ", pred_mppde1.shape)
    print("pred_mppde4.shape : ", pred_mppde4.shape)
    print("pred_mppde16.shape : ", pred_mppde16.shape)

    pred0 = (pred_mppde0)
    pred1 = (pred_mppde1)
    pred4 = (pred_mppde4)
    pred16 = (pred_mppde16)

    title = f'ns-upsampling-{sub_tr}-64to256_graph.png'
    plot_baselines(plot_dir, title, b0, b1, b4, b16, pred0,
                   pred1, pred4, pred16, x, y, time2show, baselines)
    title = f'ns-upsampling-{sub_tr}-64to256_true_graph.gif'
    gif_baselines(plot_dir, title, b0, b1, b4, b16, x, y)
    title = f'ns-upsampling-{sub_tr}-64to256_mppde.gif'
    gif_baselines(plot_dir, title, pred_mppde0, pred_mppde1,
                  pred_mppde4, pred_mppde16, x, y)


if __name__ == "__main__":
    main()
