
# Copyright 2022 Yuan Yin & Matthieu Kirchmeyer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import wandb
import einops
import hydra
import torch
from ode_model import Decoder, Derivative
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torchdiffeq import odeint
from pathlib import Path
from utils import scheduling

from eval_dino_armand import eval_dino
from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, KEY_TO_INDEX
from template.ode_dynamics import DetailedMSE


@hydra.main(config_path="config/", config_name="dino.yaml")
def main(cfg: DictConfig) -> None:

    print("cfg : ", cfg)

    #######################################################################################
    #################################### Load config  #####################################
    #######################################################################################

    checkpoint_path = cfg.optim.checkpoint_path

    cuda = torch.cuda.is_available()
    if cuda:
        gpu_id = torch.cuda.current_device()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
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

    # inr
    model_type = cfg.inr.model_type
    code_dim = cfg.inr.code_dim
    # depth of mfn (?) TODO : 4 en dur dans l'archi network
    hidden_c_enc = cfg.inr.hidden_dim
    n_layers = cfg.inr.depth

    # dynamic model
    hidden_c = cfg.dynamic.hidden_dim  # depth of mlp
    method = cfg.dynamic.method

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.learning_rate
    epsilon = cfg.optim.teacher_forcing_init
    epsilon_t = cfg.optim.teacher_forcing_decay
    epsilon_freq = cfg.optim.teacher_forcing_update
    n_epochs = cfg.optim.epochs
    n_steps = cfg.inr.n_steps  # 300 if ns 500 if sw or wave

    assert n_epochs > 1000, '1000 epoch needed to save model'

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

    print("id", run.id)
    print("dir", run.dir)

    if data_to_encode is not None:
        RESULTS_DIR = (
            Path(os.getenv("WANDB_DIR")) /
            dataset_name / data_to_encode / "dino"
        )
    else:
        RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "dino"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    if data_to_encode == None:
        run.tags = ("dino",) + (model_type,) + \
            (dataset_name,) + (f"sub={sub_tr}",)
    else:
        run.tags = (
            ("dino",)
            + (model_type,)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
        )

    if dataset_name == 'shallow-water-dino':
        multichannel = True
    else:
        multichannel = False

    #######################################################################################
    #################################### Load data ########################################
    #######################################################################################

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
    if u_train_ext is not None:
        print(
            f"data: {dataset_name}, u_train_ext: {u_train_ext.shape}, u_test_ext: {u_test_ext.shape}")
        print(
            f"grid: grid_tr_ext: {grid_tr_ext.shape}, grid_te_ext: {grid_te_ext.shape}")

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

    n_seq_train = u_train.shape[0]  # 512 en dur
    n_seq_test = u_test.shape[0]  # 512 en dur
    spatial_size = u_train.shape[1]  # 64 en dur
    state_dim = u_train.shape[2]  # N, XY, C, T
    coord_dim = grid_tr.shape[2]  # N, XY, C, T
    T = u_train.shape[-1]

    ntrain = u_train.shape[0]  # int(u_train.shape[0]*T)
    ntest = u_test.shape[0]  # int(u_test.shape[0]*T)

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, code_dim, dataset_name, data_to_encode
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, code_dim, dataset_name, data_to_encode
    )
    if u_train_ext is not None:
        trainset_ext = TemporalDatasetWithCode(
            u_train_ext, grid_tr_ext, code_dim, dataset_name, data_to_encode)
    if u_test_ext is not None:
        testset_ext = TemporalDatasetWithCode(
            u_test_ext, grid_te_ext, code_dim, dataset_name, data_to_encode)

    # create torch dataset
    train_loader = torch.utils.data.DataLoader(
        trainset,
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
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )
    if u_test_ext is not None:
        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True,
            num_workers=1,
        )

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

    T = u_train.shape[-1]
    if u_test_ext is not None:
        T_EXT = u_test_ext.shape[-1]

    dt = 1
    timestamps_train = torch.arange(0, T, dt).float().cuda()
    timestamps_ext = torch.arange(0, T_EXT, dt).float().cuda()

    #######################################################################################
    #################################### Init Networks ####################################
    #######################################################################################

    if checkpoint_path is None:  # Start from scratch
        # Decoder
        net_dec_params = {
            "state_c": state_dim,
            "code_c": code_dim,
            "hidden_c": hidden_c_enc,
            "n_layers": n_layers,
            "coord_dim": coord_dim,
            "model": model_type,
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
                    torch.zeros(sequence_length_in, code_dim *
                                state_dim).to(device)
                )
                for _ in range(n_seq_train)
            ]
        )

        print(dict(net_dec.named_parameters()).keys())
        print(dict(net_dyn.named_parameters()).keys())

        net_dec = net_dec.to(device)
        net_dyn = net_dyn.to(device)
    else:  # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu_id}")

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

    criterion = nn.MSELoss()

    # TODO : rgd lr
    optim_net_dec = torch.optim.Adam(
        [{"params": net_dec.parameters(), "lr": lr}])
    optim_net_dyn = torch.optim.Adam(
        [{"params": net_dyn.parameters(), "lr": lr / 10}])
    optim_states = torch.optim.Adam([{"params": states_params, "lr": lr / 10}])

    #######################################################################################
    #################################### Training #########################################
    #######################################################################################

    loss_tr_min, loss_ts_min, loss_relative_min = (
        float("inf"),
        float("inf"),
        float("inf"),
    )
    for epoch in range(n_epochs):

        step_show = epoch % 100 == 0
        result_dic = {}

        if epoch % epsilon_freq == 0:
            epsilon_t = epsilon_t * epsilon

        # Update Decoder and Dynamics
        if epoch != 0:
            optim_net_dec.step()
            optim_net_dec.zero_grad()

            optim_net_dyn.step()
            optim_net_dyn.zero_grad()

        code_train_loss = 0
        dec_train_loss = 0

        for i, (images, _, coords, idx) in enumerate(train_loader):

            # flatten spatial dims
            ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
            model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')

            # permute axis for forward
            ground_truth = torch.permute(
                ground_truth, (0, 3, 1, 2)).to(device)  # [B, XY, C, T] -> [B, T, XY, C]
            model_input = torch.permute(
                model_input, (0, 3, 1, 2)).to(device)  # ([B, XY, C, T] -> -> [B, T, XY, C]

            # ground_truth, model_input are B, T, X, Y, C
            b_size, t_size, hw_size, channels = ground_truth.shape

            # take only one grid cf 10 lignes plus bas
            # T = horizon d'entraînement
            index = idx.to(device)

            # Update latent states
            states_params_index = torch.stack(
                [states_params[d] for d in index], dim=1)  # T, B, L
            states = states_params_index.permute(1, 0, 2).view(
                b_size, t_size, state_dim, code_dim
            )  # B, T, C, L

            model_input_exp = model_input.view(
                b_size, t_size, hw_size, 1, coord_dim
            ).expand(b_size, t_size, hw_size, state_dim, coord_dim)  # B, T, XY, C, grid

            # B, T, XY, C
            model_output, _ = net_dec(model_input_exp, states)

            loss_l2 = criterion(
                model_output, ground_truth
            )
            optim_states.zero_grad(True)
            loss_l2.backward()
            optim_states.step()

            # Cumulate gradient of dynamics
            # t = 0..19 dans le cas extrapolation
            codes = scheduling(
                odeint,
                net_dyn,
                states_params_index.detach().clone(),
                timestamps_train,
                epsilon_t,
                method=method,
            )

            loss_l2_states = criterion(
                codes, states_params_index.detach().clone())
            loss_l2_states.backward()

            code_train_loss += loss_l2_states.item() * b_size
            dec_train_loss += loss_l2.item() * b_size

        optimize_tr = code_train_loss / n_seq_train

        tr_dic = {'code_train_loss': code_train_loss / n_seq_train,
                  'dec_train_loss': dec_train_loss / n_seq_train, }
        result_dic.update(tr_dic)

        eval_dic = {}
        if step_show:
            print("Evaluating train...")
            if u_train_ext is not None:
                # gts/mos pour plotting (ground_truths/model_ouputs)
                (
                    pred_train_mse,
                    pred_train_inter_mse,
                    pred_train_extra_mse,
                    detailed_train_eval_mse
                ) = eval_dino(
                    train_loader_ext,
                    net_dyn,
                    net_dec,
                    device,
                    method,
                    criterion,
                    state_dim,
                    code_dim,
                    coord_dim,
                    detailed_train_eval_mse,
                    timestamps_ext,
                    n_seq_train,
                    sequence_length_in,
                    sequence_length_out,
                    states_params,
                    n_steps=n_steps,
                    multichannel=False,
                    save_best=True
                )
                eval_dic.update({'pred_train_mse': pred_train_mse,
                                 'pred_train_inter_mse': pred_train_inter_mse,
                                 'pred_train_extra_mse': pred_train_extra_mse,
                                 'detailed_train_eval_mse': detailed_train_eval_mse, })

            # Out-of-domain evaluation
            print("Evaluating test...")
            if u_test_ext is not None:
                (
                    pred_test_mse,
                    pred_test_inter_mse,
                    pred_test_extra_mse,
                    detailed_test_eval_mse
                ) = eval_dino(
                    test_loader_ext,
                    net_dyn,
                    net_dec,
                    device,
                    method,
                    criterion,
                    state_dim,
                    code_dim,
                    coord_dim,
                    detailed_test_mse,
                    timestamps_ext,
                    n_seq_test,
                    sequence_length_in,
                    sequence_length_out,
                    states_params,
                    lr,
                    multichannel=multichannel,
                    n_steps=n_steps,
                )
                eval_dic.update({'pred_test_mse': pred_test_mse,
                                 'pred_test_inter_mse': pred_test_inter_mse,
                                 'pred_test_extra_mse': pred_test_extra_mse,
                                 'detailed_test_eval_mse': detailed_test_eval_mse, })

        result_dic.update(eval_dic)

        wandb.log(result_dic)

        if epoch >= 1000:
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
                    f"{RESULTS_DIR}/{run_name}_tr.pt",
                )


if __name__ == "__main__":
    main()
