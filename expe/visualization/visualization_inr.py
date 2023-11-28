import torch
import torch.nn as nn
import einops
from pathlib import Path
import os
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import DictConfig, OmegaConf
import omegaconf
import hydra

from coral.utils.data.load_data import get_dynamics_data, set_seed
from expe.load_models.load_models_inr import load_coral, load_dino
from expe.forwards.forwards_inr import forward_coral, forward_dino
from expe.config.run_names import RUN_NAMES
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode
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

    inr_run_name = RUN_NAMES[sub_from][sub_tr]["coral"]["inr"]
    dyn_run_name = RUN_NAMES[sub_from][sub_tr]["coral"]["dyn"]
    dino_run_name = RUN_NAMES[sub_from][sub_tr]["dino"]

    n_baselines = (inr_run_name != None) + \
        (dino_run_name != None)

    print("inr_run_name : ", inr_run_name)
    print("dyn_run_name : ", dyn_run_name)
    print("dino_run_name : ", dino_run_name)
    print(f"running on {n_baselines} baselines")

    baselines = []
    if dyn_run_name is not None:
        cfg_coral_dyn = torch.load(
            root_dir / "model" / f"{dyn_run_name}.pt")['cfg']
        baselines += ['CORAL']
    if inr_run_name is not None:
        cfg_coral_inr = torch.load(
            root_dir / "inr" / f"{inr_run_name}.pt")['cfg']
        baselines += ['DINO']

    upsamplings = ['0', '1', '2', '4']

    # data
    ntrain = cfg_coral_dyn.data.ntrain
    ntest = cfg_coral_dyn.data.ntest
    data_to_encode = cfg_coral_dyn.data.data_to_encode
    sub_tr = cfg_coral_dyn.data.sub_tr
    sub_te = cfg_coral_dyn.data.sub_te if sub_te is not None else sub_tr
    try:
        sub_from = cfg_coral_dyn.data.sub_from
    except omegaconf.errors.ConfigAttributeError:
        sub_from = sub_tr  # firsts runs don't have a sub_from attribute ie run 4 / 1-1
        sub_tr = 1
        sub_te = 1
    seed = cfg_coral_dyn.data.seed
    same_grid = cfg_coral_dyn.data.same_grid
    setting = cfg_coral_dyn.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg_coral_dyn.data.sequence_length_in
    sequence_length_out = cfg_coral_dyn.data.sequence_length_out

    assert sub_from == cfg.data.sub_from, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_from} but model trained on {sub_from}"
    assert sub_tr == cfg.data.sub_tr, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_tr} but model trained on {sub_tr}"
    assert sub_te == cfg.data.sub_te, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_te} but model trained on {sub_te}"

    print(
        f"running from setting {setting} with sampling {sub_from} / {sub_tr} - {sub_te}")

    # dino
    n_steps = 300
    lr_adapt = 0.005

    # coral
    code_dim_coral = cfg_coral_inr.inr.latent_dim
    width_dyn_coral = cfg_coral_dyn.dynamics.width
    depth_dyn_coral = cfg_coral_dyn.dynamics.depth
    inner_steps = cfg_coral_inr.optim.inner_steps

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

    n_seq_train = u_train.shape[0]  # 512 en dur
    n_seq_test = u_test.shape[0]  # 512 en dur
    spatial_size = u_train.shape[1]  # 64 en dur
    state_dim = u_train.shape[2]  # N, XY, C, T
    coord_dim = grid_tr.shape[2]  # N, XY, C, T
    T = u_train.shape[-1]

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, code_dim_coral, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, code_dim_coral, dataset_name, data_to_encode
    )
    if u_train_ext is not None:
        trainset_ext = TemporalDatasetWithCode(
            u_train_ext, grid_tr_ext, code_dim_coral, dataset_name, data_to_encode)
    if u_test_ext is not None:
        testset_ext = TemporalDatasetWithCode(
            u_test_ext, grid_te_ext, code_dim_coral, dataset_name, data_to_encode)
    if u_train_up1 is not None:
        trainset_up1 = TemporalDatasetWithCode(
            u_train_up1, grid_tr_up1, code_dim_coral, dataset_name, data_to_encode)
    if u_test_up1 is not None:
        testset_up1 = TemporalDatasetWithCode(
            u_test_up1, grid_te_up1, code_dim_coral, dataset_name, data_to_encode)
    if u_train_up4 is not None:
        trainset_up4 = TemporalDatasetWithCode(
            u_train_up4, grid_tr_up4, code_dim_coral, dataset_name, data_to_encode)
    if u_test_up4 is not None:
        testset_up4 = TemporalDatasetWithCode(
            u_test_up4, grid_te_up4, code_dim_coral, dataset_name, data_to_encode)
    if u_train_up16 is not None:
        trainset_up16 = TemporalDatasetWithCode(
            u_train_up16, grid_tr_up16, code_dim_coral, dataset_name, data_to_encode)
    if u_test_up16 is not None:
        testset_up16 = TemporalDatasetWithCode(
            u_test_up16, grid_te_up16, code_dim_coral, dataset_name, data_to_encode)

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=1,
        drop_last=True,
    )
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_ext is not None:
        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up1 is not None:
        train_loader_up1 = torch.utils.data.DataLoader(
            trainset_up1,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up1 is not None:
        test_loader_up1 = torch.utils.data.DataLoader(
            testset_up1,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up4 is not None:
        train_loader_up4 = torch.utils.data.DataLoader(
            trainset_up4,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up4 is not None:
        test_loader_up4 = torch.utils.data.DataLoader(
            testset_up4,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_train_up16 is not None:
        train_loader_up16 = torch.utils.data.DataLoader(
            trainset_up16,
            batch_size=batch_size_val,
            shuffle=False,
            num_workers=1,
            drop_last=True,
        )
    if u_test_up16 is not None:
        test_loader_up16 = torch.utils.data.DataLoader(
            testset_up16,
            batch_size=batch_size_val,
            shuffle=False,
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

    inr, alpha, dyn, z_mean, z_std = load_coral(root_dir, inr_run_name, dyn_run_name, data_to_encode, input_dim,
                                                output_dim, trainset, testset, multichannel, code_dim_coral, width_dyn_coral, depth_dyn_coral, inner_steps)
    net_dec, net_dyn, states_params, code_dim_dino = load_dino(
        root_dir, dino_run_name)

    torch.set_printoptions(precision=10)

    batch = next(iter(test_loader_ext))
    idx = batch[3]
    b0 = batch[0]
    x = batch[2][0, ..., 0, 0]  # XY
    y = batch[2][0, ..., 1, 0]  # XY
    pred_coral0 = forward_coral(inr, dyn, batch, inner_steps, alpha, True,
                                timestamps_ext, z_mean, z_std, dataset_name).cpu().detach().numpy()
    pred_dino0 = forward_dino(net_dec, net_dyn, batch, n_seq_train, states_params, code_dim_dino, n_steps,
                              lr_adapt, device, criterion, timestamps_ext, save_best=True, method="rk4").cpu().detach().numpy()
    batch = next(iter(test_loader_up1))
    x1 = batch[2][0, ..., 0, 0]  # XY
    y1 = batch[2][0, ..., 1, 0]  # XY
    b1 = batch[0]
    assert batch[3] == idx
    pred_coral1 = forward_coral(inr, dyn, batch, inner_steps, alpha, True,
                                timestamps_ext, z_mean, z_std, dataset_name).cpu().detach().numpy()
    pred_dino1 = forward_dino(net_dec, net_dyn, batch, n_seq_train, states_params, code_dim_dino, n_steps,
                              lr_adapt, device, criterion, timestamps_ext, save_best=True, method="rk4").cpu().detach().numpy()
    batch = next(iter(test_loader_up4))
    b4 = batch[0]
    assert batch[3] == idx
    pred_coral4 = forward_coral(inr, dyn, batch, inner_steps, alpha, True,
                                timestamps_ext, z_mean, z_std, dataset_name).cpu().detach().numpy()
    pred_dino4 = forward_dino(net_dec, net_dyn, batch, n_seq_train, states_params, code_dim_dino, n_steps,
                              lr_adapt, device, criterion, timestamps_ext, save_best=True, method="rk4").cpu().detach().numpy()
    batch = next(iter(test_loader_up16))
    b16 = batch[0]
    assert batch[3] == idx
    pred_coral16 = forward_coral(inr, dyn, batch, inner_steps, alpha, True,
                                 timestamps_ext, z_mean, z_std, dataset_name).cpu().detach().numpy()
    pred_dino16 = forward_dino(net_dec, net_dyn, batch, n_seq_train, states_params, code_dim_dino, n_steps,
                               lr_adapt, device, criterion, timestamps_ext, save_best=True, method="rk4").cpu().detach().numpy()

    time2show = 30  # On prend un pas au milieu de l'extrapolation
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    print("b0.shape : ", b0.shape)
    print("b1.shape : ", b1.shape)
    print("b4.shape : ", b4.shape)
    print("b16.shape : ", b16.shape)
    print("pred_coral0.shape : ", pred_coral0.shape)  # B, XY, C, T
    print("pred_coral1.shape : ", pred_coral1.shape)
    print("pred_coral4.shape : ", pred_coral4.shape)
    print("pred_coral16.shape : ", pred_coral16.shape)
    print("pred_dino0.shape : ", pred_dino0.shape)
    print("pred_dino1.shape : ", pred_dino1.shape)
    print("pred_dino4.shape : ", pred_dino4.shape)
    print("pred_dino16.shape : ", pred_dino16.shape)


    pred0 = (pred_coral0, pred_dino0)
    pred1 = (pred_coral1, pred_dino1)
    pred4 = (pred_coral4, pred_dino4)
    pred16 = (pred_coral16, pred_dino16)

    title = f'ns-upsampling-{sub_tr}-64to256_inr.png'
    plot_baselines(plot_dir, title, b0, b1, b4, b16, pred0,
                   pred1, pred4, pred16, x, y, time2show, baselines)

    title = f'ns-upsampling-{sub_tr}-64to256_true_inr.gif'
    gif_baselines(plot_dir, title, b0, b1, b4, b16, x, y)
    title = f'ns-upsampling-{sub_tr}-64to256_coral.gif'
    gif_baselines(plot_dir, title, pred_coral0, pred_coral1,
                  pred_coral4, pred_coral16, x, y)
    title = f'ns-upsampling-{sub_tr}-64to256_dino.gif'
    gif_baselines(plot_dir, title, pred_dino0, pred_dino1,
                  pred_dino4, pred_dino16, x, y)

    title = f'ns-errors-{sub_tr}-64to256_inr.png'
    plot_errors(plot_dir, title, b0, b1, b4, b16, pred0,
                pred1, pred4, pred16, x, y, baselines)

    title = f'ns-grid-{sub_tr}.png'
    plot_grid(plot_dir + 'dino/', title, x, y, x1, y1)
    title = f'ns-grid-{sub_tr}.png'
    plot_grid(plot_dir + 'coral/', title, x, y, x1, y1)

    times2show = range(0, 40)
    title = f'ns-true-{sub_tr}'
    save_imshow(plot_dir + 'true/', title, b1, times2show, 64)

    title = f'ns-coral-{sub_tr}'
    save_imshow(plot_dir + 'coral/', title, pred1[0], times2show, 64)
    title = f'ns-dino-{sub_tr}'
    save_imshow(plot_dir + 'dino/', title, pred1[1], times2show, 64)


if __name__ == "__main__":
    main()
