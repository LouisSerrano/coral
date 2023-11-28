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
import yaml
import numpy as np

from expe.forwards.forwards_operator import forward_fno, forward_deeponet_up, forward_deeponet
from omegaconf import DictConfig, OmegaConf
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, KEY_TO_INDEX
from expe.load_models.load_models_operator import load_fno, load_deeponet
from expe.config.run_names import RUN_NAMES
from deeponet.coral.eval import eval_deeponet


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

    sub_from = cfg.data.sub_from
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_tr

    deeponet_run_name = RUN_NAMES[sub_from][sub_tr]["deeponet"]
    fno_run_name = RUN_NAMES[sub_from][sub_tr]["fno"]

    n_baselines = (deeponet_run_name != "") + \
        (fno_run_name != "")

    print("fno_run_name : ", fno_run_name)
    print("deeponet_run_name : ", deeponet_run_name)
    print(f"running on {n_baselines} baselines")

    if fno_run_name != "":
        cfg_fno = torch.load(root_dir / "fno" / f"{fno_run_name}.pt")['cfg']
    if deeponet_run_name != "":
        cfg_deeponet = torch.load(root_dir / "deeponet" / f"{deeponet_run_name}_tr.pt")[
            'deeponet_params']  # inr forgot to change in deeopnet_2d_time

    upsamplings = ['0', '1', '2', '4']

    # data
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    # sub_from = cfg.data.sub_from
    # sub_tr = cfg.data.sub_tr
    # sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid
    setting = cfg.data.setting
    sequence_length_optim = None
    sequence_length_in = cfg.data.sequence_length_in
    sequence_length_out = cfg.data.sequence_length_out

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

    mean, sigma = u_train.mean(), u_train.std()

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
    srs = []
    if u_train_ext is not None:
        srs += [u_train_ext.shape[1]]
        u_train_ext = einops.rearrange(u_train_ext, 'B ... C T -> B (...) C T')
        grid_tr_ext = einops.rearrange(
            grid_tr_ext, 'B ... C T -> B (...) C T')
        u_test_ext = einops.rearrange(u_test_ext, 'B ... C T -> B (...) C T')
        grid_te_ext = einops.rearrange(
            grid_te_ext, 'B ... C T -> B (...) C T')
    if u_train_up1 is not None:
        srs += [u_train_up1.shape[1]]
        u_train_up1 = einops.rearrange(u_train_up1, 'B ... C T -> B (...) C T')
        grid_tr_up1 = einops.rearrange(
            grid_tr_up1, 'B ... C T -> B (...) C T')
        u_test_up1 = einops.rearrange(u_test_up1, 'B ... C T -> B (...) C T')
        grid_te_up1 = einops.rearrange(
            grid_te_up1, 'B ... C T -> B (...) C T')
    if u_train_up4 is not None:
        srs += [u_train_up4.shape[1]]
        u_train_up4 = einops.rearrange(u_train_up4, 'B ... C T -> B (...) C T')
        grid_tr_up4 = einops.rearrange(
            grid_tr_up4, 'B ... C T -> B (...) C T')
        u_test_up4 = einops.rearrange(u_test_up4, 'B ... C T -> B (...) C T')
        grid_te_up4 = einops.rearrange(
            grid_te_up4, 'B ... C T -> B (...) C T')
    if u_train_up16 is not None:
        srs += [u_train_up16.shape[1]]
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

    if fno_run_name != "":
        fno = load_fno(root_dir, fno_run_name).cuda()
        fno.eval()
    if deeponet_run_name != "":
        deeponet = load_deeponet(
            root_dir, deeponet_run_name, dataset_name).cuda()
        deeponet.eval()

    errors_train_fno = torch.zeros((4, 40)).cuda()
    errors_train_deeponet = torch.zeros((4, 40)).cuda()
    errors_test_fno = torch.zeros((4, 40)).cuda()
    errors_test_deeponet = torch.zeros((4, 40)).cuda()
    torch.set_printoptions(precision=10)
    save_dir = '/home/lise.leboudec/project/coral/xp/errors/'

    print('--- Evaluation on train dataset ---')
    for up, loader in enumerate([train_loader_ext, train_loader_up1, train_loader_up4, train_loader_up16]):
        print("up : ", upsamplings[up])
        sr = srs[up]
        for bidx, batch in enumerate(loader):
            images = batch[0].cuda() * sigma.cuda() + mean.cuda()
            n_samples = images.shape[0]
            i = 0
            if fno_run_name != "":
                with torch.no_grad():
                    tic = time.time()
                    pred_coral = forward_fno(
                        fno, batch, timestamps_ext, sigma, mean, spatial_res=sr)
                    tac = time.time()
                errors_train_fno[up, :] += ((pred_coral - images)
                                            ** 2).mean(0).mean(0).mean(0) * n_samples
                i += 1
            if deeponet_run_name != "":
                with torch.no_grad():
                    idx = batch[3]
                    u_input = trainset_ext[idx][0]  # 1, 4096, 1, 40
                    coords = trainset_ext[idx][2]  # 1, 4096, 2, 40
                    u_input_up = batch[0]  # 1, 16384, 2, 40
                    coords_up = batch[2]  # 1, 16384, 1, 40
                    images = batch[0].cuda()
                    tic = time.time()
                    pred_coral = forward_deeponet_up(
                        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device)
                    tac = time.time()
                    if up == 0:
                        tic = time.time()
                        pred_coral = forward_deeponet(
                            deeponet, batch, timestamps_ext, device)
                        pred_coral = einops.rearrange(
                            pred_coral, 'B T X C -> B X C T')
                        tac = time.time()
                    errors_train_deeponet[up, :] += ((pred_coral - images) ** 2).mean(
                        0).mean(0).mean(0) * n_samples
                i += 1

    errors_train_fno /= n_seq_train
    errors_train_deeponet /= n_seq_train

    title = f'errors_train_fno_{sub_tr}'
    np.savez(save_dir + title, errors_train_fno.cpu().detach().numpy())
    title = f'errors_train_deeponet_{sub_tr}'
    np.savez(save_dir + title, errors_train_deeponet.cpu().detach().numpy())

    print('--- Evaluation on test dataset ---')
    for up, loader in enumerate([test_loader_ext, test_loader_up1, test_loader_up4, test_loader_up16]):
        print("up : ", upsamplings[up])
        sr = srs[up]
        for bidx, batch in enumerate(loader):
            images = batch[0].cuda() * sigma.cuda() + mean.cuda()
            n_samples = images.shape[0]
            i = 0
            if fno_run_name != "":
                with torch.no_grad():
                    tic = time.time()
                    pred_coral = forward_fno(
                        fno, batch, timestamps_ext, sigma, mean, spatial_res=sr)
                    tac = time.time()
                errors_test_fno[up, :] += ((pred_coral - images)
                                           ** 2).mean(0).mean(0).mean(0) * n_samples
                i += 1
            if deeponet_run_name != "":
                with torch.no_grad():
                    idx = batch[3]
                    u_input = testset_ext[idx][0]  # 1, 4096, 1, 40
                    coords = testset_ext[idx][2]  # 1, 4096, 2, 40
                    u_input_up = batch[0]  # 1, 16384, 2, 40
                    coords_up = batch[2]  # 1, 16384, 1, 40
                    tic = time.time()
                    images = batch[0].cuda()
                    pred_coral = forward_deeponet_up(
                        deeponet, u_input, u_input_up, coords, coords_up, timestamps_ext, device)
                    tac = time.time()
                    if up == 0:
                        tic = time.time()
                        pred_coral = forward_deeponet(
                            deeponet, batch, timestamps_ext, device)
                        pred_coral = einops.rearrange(
                            pred_coral, 'B T X C -> B X C T')
                        tac = time.time()
                errors_test_deeponet[up, :] += ((pred_coral - images) ** 2).mean(
                    0).mean(0).mean(0) * n_samples
                i += 1

    errors_test_fno /= n_seq_test
    errors_test_deeponet /= n_seq_test

    title = f'errors_test_fno_{sub_tr}'
    np.savez(save_dir + title, errors_test_fno.cpu().detach().numpy())
    title = f'errors_test_deeponet_{sub_tr}'
    np.savez(save_dir + title, errors_test_deeponet.cpu().detach().numpy())


if __name__ == "__main__":
    main()
