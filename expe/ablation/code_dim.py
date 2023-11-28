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
import omegaconf
import json
import numpy as np

from expe.forwards.forwards_inr import forward_coral
from template.evaluate_inout import evaluate_coral_dyn
from omegaconf import DictConfig, OmegaConf
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, KEY_TO_INDEX
from expe.load_models.load_models_inr import load_coral
from expe.config.ablation_code_dim import RUN_NAMES
from template.ode_dynamics import DetailedMSE
from expe.visualization.visualization_functions import plot_codes


@hydra.main(config_path="config/", config_name="code_dim.yaml")
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

    inr_run_name32 = RUN_NAMES[32]["inr"]
    dyn_run_name32 = RUN_NAMES[32]["ode"]
    inr_run_name64 = RUN_NAMES[64]["inr"]
    dyn_run_name64 = RUN_NAMES[64]["ode"]
    inr_run_name128 = RUN_NAMES[128]["inr"]
    dyn_run_name128 = RUN_NAMES[128]["ode"]
    inr_run_name256 = RUN_NAMES[256]["inr"]
    dyn_run_name256 = RUN_NAMES[256]["ode"]
    inr_run_name512 = RUN_NAMES[512]["inr"]
    dyn_run_name512 = RUN_NAMES[512]["ode"]

    code_dims = [32, 64, 128, 256, 512]

    if dyn_run_name32 is not None:
        cfg_coral_dyn = torch.load(
            root_dir / "model" / f"{dyn_run_name32}.pt")['cfg']
    if inr_run_name32 is not None:
        cfg_coral_inr = torch.load(
            root_dir / "inr" / f"{inr_run_name32}.pt")['cfg']

    # data
    ntrain = cfg_coral_dyn.data.ntrain
    ntest = cfg_coral_dyn.data.ntest
    data_to_encode = cfg_coral_dyn.data.data_to_encode
    sub_tr = cfg_coral_dyn.data.sub_tr
    sub_te = cfg_coral_dyn.data.sub_te
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

    print(
        f"running from setting {setting} with sampling {sub_from} / {sub_tr} - {sub_te}")

    # coral
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

    if multichannel:
        detailed_train_eval_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                              dataset_name,
                                              mode="train_extra",
                                              n_trajectories=n_seq_train)
        detailed_test_mse = DetailedMSE(list(KEY_TO_INDEX[dataset_name].keys()),
                                        dataset_name,
                                        mode="test",
                                        n_trajectories=n_seq_test)
    else:
        detailed_train_eval_mse = None
        detailed_test_mse = None

    # experiments

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

    print(
        f"data: {dataset_name}, u_train: {u_train.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_te: {grid_te.shape}")
    if u_train_ext is not None:
        print(
            f"data: {dataset_name}, u_train_ext: {u_train_ext.shape}, u_test_ext: {u_test_ext.shape}")
        print(
            f"grid: grid_tr_ext: {grid_tr_ext.shape}, grid_te_ext: {grid_te_ext.shape}")

    n_seq_train = u_train.shape[0]  # 512 en dur
    n_seq_test = u_test.shape[0]  # 512 en dur
    spatial_size = u_train.shape[1]  # 64 en dur
    state_dim = u_train.shape[2]  # N, XY, C, T
    coord_dim = grid_tr.shape[2]  # N, XY, C, T
    T = u_train.shape[-1]

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset32 = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, 32, dataset_name, data_to_encode
    )
    testset32 = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, 32, dataset_name, data_to_encode
    )
    trainset64 = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, 64, dataset_name, data_to_encode
    )
    testset64 = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, 64, dataset_name, data_to_encode
    )
    trainset128 = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, 128, dataset_name, data_to_encode
    )
    testset128 = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, 128, dataset_name, data_to_encode
    )
    trainset256 = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, 256, dataset_name, data_to_encode
    )
    testset256 = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, 256, dataset_name, data_to_encode
    )
    trainset512 = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, 512, dataset_name, data_to_encode
    )
    testset512 = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, 512, dataset_name, data_to_encode
    )

    # create torch dataset
    train_loader32_ext = torch.utils.data.DataLoader(
        trainset32,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader32_ext = torch.utils.data.DataLoader(
        testset32,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    train_loader64_ext = torch.utils.data.DataLoader(
        trainset64,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader64_ext = torch.utils.data.DataLoader(
        testset64,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    train_loader128_ext = torch.utils.data.DataLoader(
        trainset128,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader128_ext = torch.utils.data.DataLoader(
        testset128,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    train_loader256_ext = torch.utils.data.DataLoader(
        trainset256,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader256_ext = torch.utils.data.DataLoader(
        testset256,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    train_loader512_ext = torch.utils.data.DataLoader(
        trainset512,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader512_ext = torch.utils.data.DataLoader(
        testset512,
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

    coral32 = load_coral(root_dir, inr_run_name32, dyn_run_name32, data_to_encode, input_dim,
                         output_dim, trainset32, testset32, multichannel, 32, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral64 = load_coral(root_dir, inr_run_name64, dyn_run_name64, data_to_encode, input_dim,
                         output_dim, trainset64, testset64, multichannel, 64, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral128 = load_coral(root_dir, inr_run_name128, dyn_run_name128, data_to_encode, input_dim,
                          output_dim, trainset128, testset128, multichannel, 128, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral256 = load_coral(root_dir, inr_run_name256, dyn_run_name256, data_to_encode, input_dim,
                          output_dim, trainset256, testset256, multichannel, 256, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral512 = load_coral(root_dir, inr_run_name512, dyn_run_name512, data_to_encode, input_dim,
                          output_dim, trainset512, testset512, multichannel, 512, width_dyn_coral, depth_dyn_coral, inner_steps)

    corals = [coral32, coral64, coral128, coral256, coral512]
    trainloaders = [train_loader32_ext, train_loader64_ext,
                    train_loader128_ext, train_loader256_ext, train_loader512_ext]
    testloaders = [test_loader32_ext, test_loader64_ext,
                   test_loader128_ext, test_loader256_ext, test_loader512_ext]

    criterion = nn.MSELoss()

    losses_train = torch.zeros(5, 2).cuda()
    times_train = torch.zeros(5).cuda()
    losses_test = torch.zeros(5, 2).cuda()
    times_test = torch.zeros(5).cuda()
    torch.set_printoptions(precision=10)

    criterion = nn.MSELoss()
    print('--- Evaluation on train dataset ---')
    for code, elts in enumerate(zip(corals, trainloaders, testloaders)):
        coral, train_loader, test_loader = elts

        print("code dim : ", code_dims[code])
        for bidx, batch in enumerate(train_loader):
            images = batch[0].cuda()
            n_samples = images.shape[0]
            i = 0
            inr, alpha, dyn, z_mean, z_std = coral
            tic = time.time()
            pred_coral = forward_coral(
                inr, dyn, batch, inner_steps, alpha, True, timestamps_ext, z_mean, z_std, dataset_name)
            tac = time.time()
            losses_train[code, 0] += criterion(pred_coral[..., :sequence_length_in],
                                               images[..., :sequence_length_in]) * n_samples
            losses_train[code, 1] += criterion(pred_coral[..., sequence_length_in:sequence_length_in+sequence_length_out],
                                               images[..., sequence_length_in:sequence_length_in+sequence_length_out]) * n_samples
            times_train[code] += (tac-tic)
        print(
            f"CORAL train code_dim {code_dims[code]} in-t: {losses_train[code, 0]/ n_seq_train}")
        print(
            f"CORAL train code_dim {code_dims[code]} out-t: {losses_train[code, 1]/ n_seq_train}")
        print(
            f"CORAL train code_dim {code_dims[code]} inference time for batch_size {batch_size}: {times_train[code] / n_seq_train}")

        for bidx, batch in enumerate(test_loader):
            images = batch[0].cuda()
            n_samples = images.shape[0]
            i = 0
            inr, alpha, dyn, z_mean, z_std = coral
            tic = time.time()
            pred_coral = forward_coral(
                inr, dyn, batch, inner_steps, alpha, True, timestamps_ext, z_mean, z_std, dataset_name)
            tac = time.time()
            losses_test[code, 0] += criterion(pred_coral[..., :sequence_length_in],
                                              images[..., :sequence_length_in]) * n_samples
            losses_test[code, 1] += criterion(pred_coral[..., sequence_length_in:sequence_length_in+sequence_length_out],
                                              images[..., sequence_length_in:sequence_length_in+sequence_length_out]) * n_samples
            times_test[code] += (tac-tic)
        print(
            f"CORAL test code_dim {code_dims[code]} in-t: {losses_test[code, 0]/ n_seq_test}")
        print(
            f"CORAL test code_dim {code_dims[code]} out-t: {losses_test[code, 1]/ n_seq_test}")
        print(
            f"CORAL test code_dim {code_dims[code]} inference time for batch_size {batch_size}: {times_test[code] / n_seq_test}")

        # loss_in, loss_out, loss = evaluate_coral_dyn()

    losses_train = losses_train.cpu() / n_seq_train
    times_train = times_train.cpu().numpy().astype(np.float64) / n_seq_train

    losses_test = losses_test.cpu() / n_seq_test
    times_test = times_test.cpu().numpy().astype(np.float64) / n_seq_test

    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    title = f'codes-loss-in-train.png'
    plot_codes(plot_dir, title, code_dims, losses_train[:, 0])
    title = f'codes-loss-out-train.png'
    plot_codes(plot_dir, title, code_dims, losses_train[:, 1])
    title = f'codes-loss-in-test.png'
    plot_codes(plot_dir, title, code_dims, losses_test[:, 0])
    title = f'codes-loss-out-test.png'
    plot_codes(plot_dir, title, code_dims, losses_test[:, 1])
    title = f'codes-times-train.png'
    plot_codes(plot_dir, title, code_dims, times_train)
    title = f'codes-times-test.png'
    plot_codes(plot_dir, title, code_dims, times_test)


if __name__ == "__main__":
    main()
