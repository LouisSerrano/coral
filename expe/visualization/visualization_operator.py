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
from expe.load_models.load_models_operator import load_fno, load_deeponet
from expe.forwards.forwards_operator import forward_fno, forward_deeponet_up
from expe.config.run_names import RUN_NAMES
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode
from expe.visualization.visualization_functions import plot_baselines, gif_baselines, plot_errors, save_imshow, save_scatter, plot_grid


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

    deeponet_run_name = RUN_NAMES[sub_from][sub_tr]["deeponet"]
    fno_run_name = RUN_NAMES[sub_from][sub_tr]["fno"]

    n_baselines = (deeponet_run_name != "") + \
        (fno_run_name != "")

    print("fno_run_name : ", fno_run_name)
    print("deeponet_run_name : ", deeponet_run_name)
    print(f"running on {n_baselines} baselines")

    baselines = []
    if fno_run_name != "":
        cfg_fno = torch.load(root_dir / "fno" / f"{fno_run_name}.pt")['cfg']
        baselines += ['FNO']
    if deeponet_run_name != "":
        cfg_deeponet = torch.load(root_dir / "deeponet" / f"{deeponet_run_name}_tr.pt")[
            'deeponet_params']  # inr forgot to change in deeopnet_2d_time
        baselines += ['DEEPONET']

    upsamplings = ['0', '1', '2', '4']

    # data
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te if sub_te is not None else sub_tr
    try:
        sub_from = cfg.data.sub_from
    except omegaconf.errors.ConfigAttributeError:
        sub_from = sub_tr  # firsts runs don't have a sub_from attribute ie run 4 / 1-1
        sub_tr = 1
        sub_te = 1
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    setting = cfg.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg.data.sequence_length_in
    sequence_length_out = cfg.data.sequence_length_out

    assert sub_from == cfg.data.sub_from, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_from} but model trained on {sub_from}"
    assert sub_tr == cfg.data.sub_tr, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_tr} but model trained on {sub_tr}"
    assert sub_te == cfg.data.sub_te, f"wrong run selected, sub_from asked in cfg {cfg.data.sub_te} but model trained on {sub_te}"

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

    mean, sigma = u_train.mean(), u_train.std()

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

    if fno_run_name != "":
        fno = load_fno(root_dir, fno_run_name).cuda()
        fno.eval()
    if deeponet_run_name != "":
        deeponet = load_deeponet(
            root_dir, deeponet_run_name, dataset_name).cuda()
        deeponet.eval()

    losses_train = torch.zeros((n_baselines, 4, 2)).cuda()
    losses_test = torch.zeros((n_baselines, 4, 2)).cuda()
    torch.set_printoptions(precision=10)

    batch = next(iter(test_loader_ext))
    idx = batch[3]
    b0 = batch[0]
    x = batch[2][0, ..., 0, 0]  # XY
    y = batch[2][0, ..., 1, 0]  # XY
    u_input = batch[0]  # 1, 4096, 1, 40
    coords = batch[2]  # 1, 4096, 2, 40
    u_input_up = batch[0]  # 1, 16384, 2, 40
    coords_up = batch[2]  # 1, 16384, 1, 40
    if fno_run_name != "":
        pred_fno0 = forward_fno(fno, batch, timestamps_ext,
                                sigma, mean, spatial_res=64).cpu().detach().numpy()
    pred_deeponet0 = forward_deeponet_up(
        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device).cpu().detach().numpy()

    batch = next(iter(test_loader_up1))
    b1 = batch[0]
    x1 = batch[2][0, ..., 0, 0]  # XY
    y1 = batch[2][0, ..., 1, 0]  # XY
    assert batch[3] == idx
    u_input_up = batch[0]  # 1, 16384, 2, 40
    coords_up = batch[2]  # 1, 16384, 1, 40
    if fno_run_name != "":
        pred_fno1 = forward_fno(fno, batch, timestamps_ext,
                                sigma, mean, spatial_res=64).cpu().detach().numpy()
    pred_deeponet1 = forward_deeponet_up(
        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device).cpu().detach().numpy()

    batch = next(iter(test_loader_up4))
    b4 = batch[0]
    assert batch[3] == idx
    u_input_up = batch[0]  # 1, 16384, 2, 40
    coords_up = batch[2]  # 1, 16384, 1, 40
    if fno_run_name != "":
        pred_fno4 = forward_fno(fno, batch, timestamps_ext,
                                sigma, mean, spatial_res=128).cpu().detach().numpy()
    pred_deeponet4 = forward_deeponet_up(
        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device).cpu().detach().numpy()

    batch = next(iter(test_loader_up16))
    b16 = batch[0]
    assert batch[3] == idx
    u_input_up = batch[0]  # 1, 16384, 2, 40
    coords_up = batch[2]  # 1, 16384, 1, 40
    if fno_run_name != "":
        pred_fno16 = forward_fno(fno, batch, timestamps_ext,
                                 sigma, mean, spatial_res=256).cpu().detach().numpy()
    pred_deeponet16 = forward_deeponet_up(
        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device).cpu().detach().numpy()
    # pred_deeponet16 = torch.zeros((1, 65536, 1, 40))

    time2show = 30
    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    # b0 = trainset_ext[idx]
    # b1 = trainset_up1[idx]
    # b4 = trainset_up4[idx]
    # b16 = trainset_up16[idx]

    print("b0.shape : ", b0.shape)
    print("b1.shape : ", b1.shape)
    print("b4.shape : ", b4.shape)
    print("b16.shape : ", b16.shape)
    if fno_run_name != "":
        print("pred_fno0.shape : ", pred_fno0.shape)  # B, XY, C, T
        print("pred_fno1.shape : ", pred_fno1.shape)
        print("pred_fno4.shape : ", pred_fno4.shape)
        print("pred_fno16.shape : ", pred_fno16.shape)
    print("pred_deeponet0.shape : ", pred_deeponet0.shape)  # B, XY, C, T
    print("pred_deeponet1.shape : ", pred_deeponet1.shape)
    print("pred_deeponet4.shape : ", pred_deeponet4.shape)
    print("pred_deeponet16.shape : ", pred_deeponet16.shape)

    if fno_run_name != "":
        pred0 = (pred_fno0, pred_deeponet0)
        pred1 = (pred_fno1, pred_deeponet1)
        pred4 = (pred_fno4, pred_deeponet4)
        pred16 = (pred_fno16, pred_deeponet16)
    else:
        pred0 = (pred_deeponet0, )
        pred1 = (pred_deeponet1, )
        pred4 = (pred_deeponet4, )
        pred16 = (pred_deeponet16, )

    title = f'ns-upsampling-{sub_tr}-64to256_operator.png'
    plot_baselines(plot_dir, title, b0, b1, b4, b16, pred0,
                   pred1, pred4, pred16, x, y, time2show, baselines)

    title = f'ns-upsampling-{sub_tr}-64to256_true_operator.gif'
    gif_baselines(plot_dir, title, b0, b1, b4, b16, x, y)
    if fno_run_name != "":
        title = f'ns-upsampling-{sub_tr}-64to256_fno.gif'
        gif_baselines(plot_dir, title, pred_fno0, pred_fno1,
                      pred_fno4, pred_fno16, x, y)
    title = f'ns-upsampling-{sub_tr}-64to256_deeponet.gif'
    gif_baselines(plot_dir, title, pred_deeponet0, pred_deeponet1,
                  pred_deeponet4, pred_deeponet16, x, y)

    title = f'ns-errors-{sub_tr}-64to256_operator.png'
    plot_errors(plot_dir, title, b0, b1, b4, b16, pred0,
                pred1, pred4, pred16, x, y, baselines)

    title = f'ns-grid-{sub_tr}.png'
    plot_grid(plot_dir + 'deeponet/', title, x, y, x1, y1)

    times2show = range(0, 40)
    title = f'ns-true-{sub_tr}'
    save_imshow(plot_dir, title, b1, times2show, 64)
    if fno_run_name != "":
        title = f'ns-grid-{sub_tr}.png'
        plot_grid(plot_dir + 'fno/', title, x, y, x1, y1)
        title = f'ns-fno-{sub_tr}'
        save_imshow(plot_dir + 'fno/', title, pred1[0], times2show, 64)
        title = f'ns-deeponet-{sub_tr}'
        save_imshow(plot_dir + 'deeponet/', title, pred1[1], times2show, 64)
    else:
        title = f'ns-deeponet-{sub_tr}'
        save_imshow(plot_dir + 'deeponet+', title, pred1[0], times2show, 64)


if __name__ == "__main__":
    main()
