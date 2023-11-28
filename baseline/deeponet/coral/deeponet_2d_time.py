import os
from torch import nn
import torch
import hydra
import wandb
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import einops

from coral.utils.data.load_data import (set_seed, get_dynamics_data)
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode, KEY_TO_INDEX
from template.ode_dynamics import DetailedMSE
from deeponet.coral.deeponet_model import DeepONet, AR_forward
from eval import eval_deeponet

@hydra.main(config_path="config/", config_name="deeponet.yaml")
def main(cfg: DictConfig) -> None:

    checkpoint_path = cfg.optim.checkpoint_path
    print(torch.__version__)

    cuda = torch.cuda.is_available()
    if cuda:
        gpu_id = torch.cuda.current_device()
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    print("device : ", device)

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

    # deeponet
    model_type = cfg.deeponet.model_type
    code_dim = 1
    branch_depth = cfg.deeponet.branch_depth
    trunk_depth = cfg.deeponet.trunk_depth
    width = cfg.deeponet.width

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.learning_rate
    n_epochs = cfg.optim.epochs

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

    if dataset_name == 'shallow-water-dino':
        multichannel = True
    else:
        multichannel = False

    run = wandb.init(entity=entity, project=project)
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    run_name = wandb.run.name
    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "deeponet"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    if data_to_encode == None:
        run.tags = ("deeponet",) + (model_type,) + \
            (dataset_name,) + (f"sub={sub_tr}",)
    else:
        run.tags = (
            ("deeponet",)
            + (model_type,)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
        )

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
        shuffle=True, # TODO : here shuffle to False because error cuda (?!)
        num_workers=1,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True, # TODO : here shuffle to False because error cuda (?!)
        num_workers=1,
    )
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size_val,
            shuffle=True, # TODO : here shuffle to False because error cuda (?!)
            num_workers=1,
        )
    if u_test_ext is not None:
        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext,
            batch_size=batch_size_val,
            shuffle=True, # TODO : here shuffle to False because error cuda (?!)
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

    # Forecaster
    if checkpoint_path is None:
        net_dyn_params = {
            'branch_dim': spatial_size,
            'branch_depth': branch_depth,
            'trunk_dim': coord_dim,
            'trunk_depth': trunk_depth,
            'width': width
        } # TODO : re - faire arguments 
        deeponet = DeepONet(**net_dyn_params, logger=None, input_dataset=dataset_name)
    else : 
        checkpoint = torch.load(checkpoint_path, map_location=f"cuda:{gpu_id}")
        # TODO

    # init_weights(net_dyn, init_type='orthogonal', init_gain=0.2)

    print(dict(deeponet.named_parameters()).keys())

    deeponet = deeponet.to(device)
    criterion = nn.MSELoss()
    optim_deeponet = torch.optim.Adam([{'params': deeponet.parameters(), 'lr': lr, 'weight_decay': 1e-7}])

    # Train
    loss_tr_min = float('inf')
    for epoch in range(n_epochs):
        print("epoch : ", epoch)
        step_show = epoch % 100 == 0
        result_dic = {}

        if epoch != 0:
            optim_deeponet.step()
            optim_deeponet.zero_grad()

        train_loss = 0
        
        for i, (images, _, coords, idx) in enumerate(train_loader):
            # flatten spatial dims
            t = timestamps_train.to(device)
            ground_truth = einops.rearrange(images, 'B ... C T -> B (...) C T')
            model_input = einops.rearrange(coords, 'B ... C T -> B (...) C T')

            # permute axis for forward
            ground_truth = torch.permute(
                ground_truth, (0, 3, 1, 2)).to(device)  # [B, XY, C, T] -> [B, T, XY, C]
            model_input = torch.permute(
                model_input, (0, 3, 1, 2))[:, 0, :, :].to(device)  # ([B, XY, C, T] -> -> [B, T, XY, C] -> [B, XY, C]
            # On prend que la première grille (c'est tjs la mm dans deeponet) 
            b_size, t_size, hw_size, channels = ground_truth.shape

            # t is T, model_input is B, T, XY, grid, ground_truth is B, T, XY, C

            model_output = AR_forward(deeponet, t, model_input, ground_truth) 
            loss_l2_states = criterion(model_output, ground_truth)
            loss_opt_states = loss_l2_states

            optim_deeponet.zero_grad()
            loss_opt_states.backward()
            optim_deeponet.step()

            train_loss += loss_l2_states.item() * b_size
        optimize_tr = train_loss

        tr_dic = {'train_loss': train_loss / n_seq_train,
        }       
        result_dic.update(tr_dic)

        eval_dic = {}
        if step_show:
            print(f"train_loss at epoch {epoch}: ", train_loss / n_seq_train)

            print("Evaluating train...")
            if u_train_ext is not None:
                # gts/mos pour plotting (ground_truths/model_ouputs)
                (
                    pred_train_mse,
                    pred_train_inter_mse,
                    pred_train_extra_mse
                ) = eval_deeponet(
                    deeponet, 
                    train_loader_ext, 
                    device, 
                    timestamps_ext,
                    criterion, 
                    n_seq_train, 
                    sequence_length_in, 
                    sequence_length_out, 
                    detailed_train_eval_mse
                )
                eval_dic.update({'pred_train_mse': pred_train_mse,
                                 'pred_train_inter_mse': pred_train_inter_mse,
                                 'pred_train_extra_mse': pred_train_extra_mse,})

            # Out-of-domain evaluation
            print("Evaluating test...")
            if u_test_ext is not None:
                (
                    pred_test_mse,
                    pred_test_inter_mse,
                    pred_test_extra_mse
                ) = eval_deeponet(
                    deeponet, 
                    test_loader_ext, 
                    device, 
                    timestamps_ext,
                    criterion, 
                    n_seq_test, 
                    sequence_length_in, 
                    sequence_length_out, 
                    detailed_test_mse
                )
                eval_dic.update({'pred_test_mse': pred_test_mse,
                                 'pred_test_inter_mse': pred_test_inter_mse,
                                 'pred_test_extra_mse': pred_test_extra_mse,})

        result_dic.update(eval_dic)

        wandb.log(result_dic)

        if loss_tr_min > optimize_tr:
            loss_tr_min = optimize_tr
            torch.save(
                {
                    "epoch": epoch,
                    "deeponet_state_dict": deeponet.state_dict(),
                    "loss_in_test": loss_tr_min,
                    "deeponet_params": net_dyn_params,
                    "cfg": cfg,
                },
                f"{RESULTS_DIR}/{run_name}_tr.pt",
            )


if __name__ == "__main__":
    main()

