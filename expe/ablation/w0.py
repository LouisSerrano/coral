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
from expe.config.ablation_w0 import RUN_NAMES
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

    inr_run_name10 = RUN_NAMES[10]["inr"]
    dyn_run_name10 = RUN_NAMES[10]["ode"]
    inr_run_name20 = RUN_NAMES[20]["inr"]
    dyn_run_name20 = RUN_NAMES[20]["ode"]
    inr_run_name30 = RUN_NAMES[30]["inr"]
    dyn_run_name30 = RUN_NAMES[30]["ode"]

    w0s = [10, 20, 30]

    if dyn_run_name10 is not None:
        cfg_coral_dyn = torch.load(
            root_dir / "model" / f"{dyn_run_name10}.pt")['cfg']
    if inr_run_name10 is not None:
        cfg_coral_inr = torch.load(
            root_dir / "inr" / f"{inr_run_name10}.pt")['cfg']

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

    # dino
    n_steps = 300
    lr_adapt = 0.005

    # coral
    width_dyn_coral = cfg_coral_dyn.dynamics.width
    depth_dyn_coral = cfg_coral_dyn.dynamics.depth
    inner_steps = cfg_coral_inr.optim.inner_steps
    code_dim = cfg_coral_inr.inr.latent_dim

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

    trainset = TemporalDatasetWithCode(
        u_train_ext, grid_tr_ext, code_dim, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test_ext, grid_te_ext, code_dim, dataset_name, data_to_encode
    )

    # create torch dataset
    train_loader_ext = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    test_loader_ext = torch.utils.data.DataLoader(
        testset,
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

    coral10 = load_coral(root_dir, inr_run_name10, dyn_run_name10, data_to_encode, input_dim,
                         output_dim, trainset, testset, multichannel, code_dim, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral20 = load_coral(root_dir, inr_run_name20, dyn_run_name20, data_to_encode, input_dim,
                         output_dim, trainset, testset, multichannel, code_dim, width_dyn_coral, depth_dyn_coral, inner_steps)
    coral30 = load_coral(root_dir, inr_run_name30, dyn_run_name30, data_to_encode, input_dim,
                         output_dim, trainset, testset, multichannel, code_dim, width_dyn_coral, depth_dyn_coral, inner_steps)

    corals = [coral10, coral20, coral30]

    criterion = nn.MSELoss()

    losses_train = torch.zeros(3, 2).cuda()
    times_train = torch.zeros(3).cuda()
    losses_test = torch.zeros(3, 2).cuda()
    times_test = torch.zeros(3).cuda()
    torch.set_printoptions(precision=10)

    criterion = nn.MSELoss()
    print('--- Evaluation on train dataset ---')
    for code, elts in enumerate(corals):
        coral = elts

        print("w0: ", w0s[code])
        for bidx, batch in enumerate(train_loader_ext):
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
            f"CORAL train code_dim {w0s[code]} in-t: {losses_train[code, 0]/ n_seq_train}")
        print(
            f"CORAL train code_dim {w0s[code]} out-t: {losses_train[code, 1]/ n_seq_train}")
        print(
            f"CORAL train code_dim {w0s[code]} inference time for batch_size {batch_size}: {times_train[code] / n_seq_train}")

        for bidx, batch in enumerate(test_loader_ext):
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
            f"CORAL test code_dim {w0s[code]} in-t: {losses_test[code, 0]/ n_seq_test}")
        print(
            f"CORAL test code_dim {w0s[code]} out-t: {losses_test[code, 1]/ n_seq_test}")
        print(
            f"CORAL test code_dim {w0s[code]} inference time for batch_size {batch_size}: {times_test[code] / n_seq_test}")

        # loss_in, loss_out, loss = evaluate_coral_dyn()

    losses_train = losses_train.cpu() / n_seq_train
    times_train = times_train.cpu().numpy().astype(np.float64) / n_seq_train

    losses_test = losses_test.cpu() / n_seq_test
    times_test = times_test.cpu().numpy().astype(np.float64) / n_seq_test

    plot_dir = '/home/lise.leboudec/project/coral/xp/vis/'

    title = f'w0-loss-in-train.png'
    plot_codes(plot_dir, title, w0s, losses_train[:, 0])
    title = f'w0-loss-out-train.png'
    plot_codes(plot_dir, title, w0s, losses_train[:, 1])
    title = f'w0-loss-in-test.png'
    plot_codes(plot_dir, title, w0s, losses_test[:, 0])
    title = f'w0-loss-out-test.png'
    plot_codes(plot_dir, title, w0s, losses_test[:, 1])
    title = f'w0-times-train.png'
    plot_codes(plot_dir, title, w0s, times_train)
    title = f'w0-times-test.png'
    plot_codes(plot_dir, title, w0s, times_test)


if __name__ == "__main__":
    main()
