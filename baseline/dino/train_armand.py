import os
import yaml
import hydra
import einops
import torch
import wandb

from pathlib import Path
from dino.ode_model import Decoder, Derivative
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdiffeq import odeint
from dino.eval_dino_armand import eval_dino, DetailedMSE
from dino.utils import (count_parameters, scheduling)
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, KEY_TO_INDEX
from coral.utils.data.load_data import get_dynamics_data, set_seed


@hydra.main(config_path="config", config_name="dino_armand.yaml")
def main(cfg: DictConfig) -> None:
    path_checkpoint = cfg.path_checkpoint

    # data
    dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    seed = cfg.data.seed
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len
    same_grid = cfg.data.same_grid

    # optim
    lr = cfg.optim.lr
    epochs = cfg.optim.epochs
    batch_size = cfg.optim.minibatch_size
    batch_size_val = cfg.optim.minibatch_val_size

    # inr
    state_dim = cfg.inr.state_dim
    code_dim = cfg.inr.code_dim
    hidden_c_enc = cfg.inr.hidden_c_enc
    n_layers = cfg.inr.n_layers
    coord_dim = cfg.inr.coord_dim

    # forecaster
    hidden_c = cfg.forecaster.hidden_c
    epsilon = cfg.forecaster.teacher_forcing_init
    epsilon_t = cfg.forecaster.teacher_forcing_decay
    epsilon_freq = cfg.forecaster.teacher_forcing_update

    # wandb
    entity = cfg.wandb.entity
    project = cfg.wandb.project
    run_id = cfg.wandb.id
    run_name = cfg.wandb.name

    # cuda
    device = "cuda"

    # set seed
    set_seed(seed)

    if dataset_name == 'shallow-water-dino':
        multichannel = True
    else:
        multichannel = False

    (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
        dir,
        dataset_name,
        ntrain,
        ntest,
        seq_inter_len=seq_inter_len,
        seq_extra_len=seq_extra_len,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
    )

    u_train = einops.rearrange(u_train, 'N ... T -> N T ...')
    u_eval_extrapolation = einops.rearrange(
        u_eval_extrapolation, 'N ... T -> N T ...')
    u_test = einops.rearrange(u_test, 'N ... T -> N T ...')
    grid_tr = einops.rearrange(grid_tr, 'N ... T -> N T ...')
    grid_tr_extra = einops.rearrange(grid_tr_extra, 'N ... T -> N T ...')
    grid_te = einops.rearrange(grid_te, 'N ... T -> N T ...')

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, code_dim, dataset_name, None
    )

    trainset_extra = TemporalDatasetWithCode(
        u_eval_extrapolation, grid_tr_extra, code_dim, dataset_name, None
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, code_dim, dataset_name, None
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    train_extra_loader = torch.utils.data.DataLoader(
        trainset_extra,
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

    n_seq_train = u_train.shape[0]
    n_seq_test = u_test.shape[0]
    T_train = u_train.shape[1]
    T_test = u_test.shape[1]
    dt = 1

    timestamps_train = torch.arange(0, T_train, dt).float().cuda()
    timestamps_test = torch.arange(0, T_test, dt).float().cuda()

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

    method = "rk4"

    if dataset_name == "shallow-water-dino":
        n_steps = 500
    else:
        n_steps = 300

    if path_checkpoint is None:  # Start from scratch
        # Decoder
        net_dec_params = {
            "state_c": state_dim,
            "code_c": code_dim,
            "hidden_c": hidden_c_enc,
            "n_layers": n_layers,
            "coord_dim": coord_dim,
        }
        # Forecaster
        net_dyn_params = {
            "state_c": state_dim,
            "hidden_c": hidden_c,
            "code_c": code_dim,
        }
        net_dec = Decoder(**net_dec_params)
        net_dyn = Derivative(**net_dyn_params)
        states_params = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(seq_inter_len, code_dim * state_dim).to(device)
                )
                for _ in range(n_seq_train)
            ]
        )
        print(dict(net_dec.named_parameters()).keys())
        print(dict(net_dyn.named_parameters()).keys())

        net_dec = net_dec.to(device)
        net_dyn = net_dyn.to(device)

    else:  # Load checkpoint
        checkpoint = torch.load(path_checkpoint, map_location=f"cuda")
        net_dec_params = checkpoint["net_dec_params"]
        state_dim = net_dec_params["state_c"]
        code_dim = net_dec_params["code_c"]
        net_dec = Decoder(**net_dec_params)
        net_dec_dict = net_dec.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["dec_state_dict"].items() if k in net_dec_dict
        }
        net_dec_dict.update(pretrained_dict)
        net_dec.load_state_dict(net_dec_dict)
        print(dict(net_dec.named_parameters()).keys())

        net_dyn_params = checkpoint["net_dyn_params"]
        net_dyn = Derivative(**net_dyn_params)
        net_dyn_dict = net_dyn.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["dyn_state_dict"].items() if k in net_dyn_dict
        }
        net_dyn_dict.update(pretrained_dict)
        net_dyn.load_state_dict(net_dyn_dict)
        print(dict(net_dyn.named_parameters()).keys())

        states_params = checkpoint["states_params"]
        net_dec = net_dec.to(device)
        net_dyn = net_dyn.to(device)

    run_dir = (
        os.path.join(os.getenv("WANDB_DIR"), f"wandb/{cfg.wandb.dir}")
        if cfg.wandb.dir is not None
        else None
    )

    print("run dir given", run_dir)

    run = wandb.init(
        entity=entity,
        project=project,
        name=run_name,
        id=run_id,
        dir=None,
    )

    print(f"data: {dataset_name}, u_train: {u_train.shape}, u_train_eval: {u_eval_extrapolation.shape}, u_test: {u_test.shape}")
    print(
        f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}")

    if run_dir is not None:
        os.symlink(run.dir.split("/files")[0], run_dir)

    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name

    print("id", run.id)
    print("dir", run.dir)

    model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "dino" / "model"
    os.makedirs(str(model_dir), exist_ok=True)

    criterion = nn.MSELoss()

    optim_net_dec = torch.optim.Adam(
        [{"params": net_dec.parameters(), "lr": lr}])
    optim_net_dyn = torch.optim.Adam(
        [{"params": net_dyn.parameters(), "lr": lr / 10}])
    optim_states = torch.optim.Adam([{"params": states_params, "lr": lr / 10}])

    # Logs
    print(f"seed: {seed}")
    print(f"method: {method}")
    print(f"code_c: {code_dim}")
    print(f"lr: {lr}")
    print(
        f"n_params forecaster: {count_parameters(net_dec) + count_parameters(net_dyn)}")
    print(f"coord_dim: {coord_dim}")
    print(f"n_frames_train: {seq_inter_len}")

    # Train
    loss_tr_min = float("inf")

    for epoch in range(epochs):
        # Update Decoder and Dynamics
        step_show = epoch % 100 == 0
        if epoch % epsilon_freq == 0:
            epsilon_t *= epsilon

        pred_train_mse = 0
        code_train_mse = 0

        if epoch != 0:
            optim_net_dec.step()
            optim_net_dec.zero_grad()

            optim_net_dyn.step()
            optim_net_dyn.zero_grad()

        for i, (images, _, coords, idx) in enumerate(train_loader):
            ground_truth = images.to(device)
            if multichannel:
                model_input = coords.unsqueeze(-2).repeat(1,
                                                          1, 1, 1, 2, 1).to(device)
            else:
                model_input = coords.unsqueeze(-2).to(device)

            index = idx.to(device)
            b_size, t_size, _, _, _ = ground_truth.shape

            # Update latent states
            states_params_index = torch.stack(
                [states_params[d] for d in index], dim=1)
            states = states_params_index.permute(1, 0, 2).view(
                b_size, t_size, state_dim, code_dim)
            model_output, _ = net_dec(model_input, states)
            loss_l2 = criterion(model_output, ground_truth)
            optim_states.zero_grad(True)
            loss_l2.backward()
            optim_states.step()

            # Cumulate gradient of dynamics
            codes = scheduling(odeint, net_dyn, states_params_index.detach().clone(),
                               timestamps_train, epsilon_t, method=method)

            loss_l2_states = criterion(
                codes, states_params_index.detach().clone())
            loss_l2_states.backward()
            code_train_mse += loss_l2_states.item() * b_size

        code_train_mse = code_train_mse / n_seq_train

        if step_show:
            print("Evaluating train...")
            pred_train_mse, pred_train_inter_mse, pred_train_extra_mse, detailed_train_eval_mse = eval_dino(
                train_extra_loader, net_dyn, net_dec, device, method,
                criterion, state_dim, code_dim, coord_dim, detailed_train_eval_mse, timestamps_test, n_seq_train,
                seq_inter_len, seq_extra_len, states_params, multichannel=multichannel, n_steps=n_steps,
            )

            # Out-of-domain evaluation
            print("Evaluating test...")
            pred_test_mse, pred_test_inter_mse, pred_test_extra_mse, detailed_test_mse = eval_dino(
                test_loader, net_dyn, net_dec, device, method, criterion, state_dim,
                code_dim, coord_dim, detailed_test_mse, timestamps_test, n_seq_test, seq_inter_len, seq_extra_len,
                states_params, lr, multichannel=multichannel, n_steps=n_steps
            )

            optimize_tr = code_train_mse

            log_dic = {
                "pred_train_mse": pred_train_mse,
                "pred_train_inter_mse": pred_train_inter_mse,
                "pred_train_extra_mse": pred_train_extra_mse,
                "pred_test_mse": pred_test_mse,
                "pred_test_mse_inter": pred_test_inter_mse,
                "pred_test_mse_extra": pred_test_extra_mse,
                "code_train_mse": code_train_mse,
            }
            if multichannel:

                dic_train_extra_mse = detailed_train_eval_mse.get_dic()
                detailed_train_eval_mse.reset_dic()

                dic_test_mse = detailed_test_mse.get_dic()
                detailed_test_mse.reset_dic()

                log_dic.update(dic_train_extra_mse)
                log_dic.update(dic_test_mse)

            wandb.log(log_dic)
        else:
            wandb.log(
                {
                    "code_train_mse": code_train_mse,
                },
                step=epoch,
                commit=not step_show,
            )

        if loss_tr_min > optimize_tr:
            loss_tr_min = optimize_tr
            torch.save(
                {
                    "epoch": epoch,
                    "dec_state_dict": net_dec.state_dict(),
                    "dyn_state_dict": net_dyn.state_dict(),
                    "states_params": states_params,
                    "loss_in_test": loss_tr_min,
                    "net_dec_params": net_dec_params,
                    "net_dyn_params": net_dyn_params,
                },
                f"{model_dir}/{run_name}.pt",
            )
    return code_train_mse


if __name__ == "__main__":
    main()
