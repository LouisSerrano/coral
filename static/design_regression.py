import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint
from pathlib import Path
import os

from coral.losses import batch_mse_rel_fn
from coral.mfn import FourierNet, HyperMAGNET, HyperMultiscaleBACON
from coral.mlp import MLP, Derivative, ResNet
from coral.siren import ModulatedSiren
from coral.utils.data.load_data import set_seed
from coral.utils.plot import show
from coral.utils.data.load_data import get_operator_data
from coral.utils.data.operator_dataset import OperatorDataset
from coral.utils.models.load_inr import create_inr_instance
import torch.utils.checkpoint as cp
from coral.metalearning import outer_step
from torch.utils.data import DataLoader
from coral.utils.data.load_modulations import load_operator_modulations

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

@hydra.main(config_path="../config/static/", config_name="regression.yaml")
def main(cfg: DictConfig) -> None:
    torch.set_default_dtype(torch.float32)
    
    # submitit.JobEnvironment()
    # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    #sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    seed = cfg.data.seed
    same_grid = cfg.data.same_grid

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    gamma_step = cfg.optim.gamma_step
    epochs = cfg.optim.epochs

    # inr
    load_run_name = cfg.inr.run_name
    inner_steps = cfg.inr.inner_steps

    # model
    model_type = cfg.model.model_type
    hidden = cfg.model.width
    depth = cfg.model.depth
    dropout = cfg.model.dropout
    activation = cfg.model.activation
   
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

    root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name
    inr_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"
    modulations_dir = Path(os.getenv("WANDB_DIR")) / \
        dataset_name / "modulations"
    model_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "model"

    os.makedirs(str(inr_dir), exist_ok=True)
    os.makedirs(str(modulations_dir), exist_ok=True)
    os.makedirs(str(model_dir), exist_ok=True)

    # we need the latent dim and the sub_tr used for training
    input_inr = torch.load(root_dir / "inr" / f"{load_run_name}.pt")
    load_cfg = input_inr['cfg']
    latent_dim_in = input_inr["cfg"].inr_in.latent_dim
    latent_dim_out = input_inr["cfg"].inr_out.latent_dim
    seed = input_inr["cfg"].data.seed

    set_seed(seed)

    x_train, y_train, x_test, y_test, grid_tr, grid_te = get_operator_data(
    data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True)

    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)
    print('grid_tr', grid_tr.shape)
    print('grid_te', grid_te.shape)

    trainset = OperatorDataset(x_train,
        y_train,
        grid_tr,
        latent_dim_a=latent_dim_in,
        latent_dim_u=latent_dim_out,
        dataset_name=None,
        data_to_encode=None,
    )

    testset = OperatorDataset(x_test,
        y_test,
        grid_te,
        latent_dim_a=latent_dim_in,
        latent_dim_u=latent_dim_out,
        dataset_name=None,
        data_to_encode=None,
    )

    ntrain = len(trainset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    run.tags = (
        ("code-regression",) + (dataset_name,)
    )
   
    input_dim = 2
    output_dim_in = 2
    output_dim_out = 1

    # load inr weights
    load_cfg.inr = load_cfg.inr_in
    inr_in = create_inr_instance(
        load_cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda"
    )
    inr_in.load_state_dict(input_inr["inr_in"])
    inr_in.eval()
    alpha_in = input_inr['alpha_in']

    
    load_cfg.inr = load_cfg.inr_out
    inr_out = create_inr_instance(
        load_cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda"
    )
    inr_out.load_state_dict(input_inr["inr_out"])
    inr_out.eval()
    alpha_out = input_inr['alpha_out']

    # load modualations

    modulations = load_operator_modulations(
        trainset,
        testset,
        inr_in,
        inr_out,
        modulations_dir,
        load_run_name,
        inner_steps=inner_steps,
        alpha_a=alpha_in,
        alpha_u=alpha_out,
        batch_size=4,
        data_to_encode=None,
        try_reload=False)
    
    za_tr = modulations['za_train']
    za_te = modulations['za_test']
    zu_tr = modulations['zu_train']
    zu_te = modulations['zu_test']

    mu_a = za_tr.mean(0) #.mean(0)
    sigma_a = za_tr.std(0) #.std(0)
    mu_u = torch.Tensor([0])#zu_tr.mean(0) #0
    sigma_u = torch.Tensor([1]) #zu_tr.std(0) #1

    za_tr = (za_tr - mu_a) / sigma_a
    za_te = (za_te - mu_a) / sigma_a
    zu_tr = (zu_tr - mu_u) / sigma_u
    zu_te = (zu_te - mu_u) / sigma_u

    trainset.z_a = za_tr
    trainset.z_u = zu_tr
    testset.z_a = za_te
    testset.z_u = zu_te

    # create torch dataset
    train_loader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        batch_size=batch_size_val,
        shuffle=True,
        num_workers=1,
    )
    
    model = ResNet(input_dim=latent_dim_in,
                hidden_dim=hidden,
                output_dim=latent_dim_out,
                depth=depth,
                dropout=cfg.model.dropout).cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma_step,
        patience=250,
        threshold=0.01,
        threshold_mode="rel",
        cooldown=0,
        min_lr=1e-5,
        eps=1e-08,
        verbose=True,
    )

    best_loss = np.inf

    for step in range(epochs):
        pred_train_mse = 0
        pred_test_mse = 0
        code_train_mse = 0
        code_test_mse = 0
        step_show = step % 100 == 0

        for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(train_loader):  
            model.train()
            a_s = a_s.cuda()
            u_s = u_s.cuda()
            za_s = za_s.cuda()
            zu_s = zu_s.cuda()
            coords = coords.cuda()
            n_samples = a_s.shape[0]

            z_pred = model(za_s)

            loss = ((z_pred - zu_s) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            code_train_mse += loss.item() * n_samples

            if step_show:
                with torch.no_grad():
                    pred = inr_out.modulated_forward(coords, z_pred*sigma_u.cuda() + mu_u.cuda())
                pred_train_mse += batch_mse_rel_fn(pred, u_s).mean() * n_samples
             
        code_train_mse = code_train_mse / ntrain

        if step_show:
            pred_train_mse = pred_train_mse / ntrain

        scheduler.step(code_train_mse)

        if step_show:
            for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(test_loader):
                model.eval()
                a_s = a_s.cuda()
                u_s = u_s.cuda()
                za_s = za_s.cuda()
                zu_s = zu_s.cuda()
                coords = coords.cuda()
                n_samples = a_s.shape[0]

                with torch.no_grad():
                    z_pred = model(za_s)
                loss = ((z_pred - zu_s) ** 2).mean()
                code_test_mse += loss.item() * n_samples
                with torch.no_grad():
                    pred = inr_out.modulated_forward(coords, z_pred*sigma_u.cuda() + mu_u.cuda())
                pred_test_mse += batch_mse_rel_fn(pred, u_s).mean() * n_samples

            code_test_mse = code_test_mse / ntest
            pred_test_mse = pred_test_mse / ntest

        if step_show:
            log_dic = {
                "pred_test_mse": pred_test_mse,
                "pred_train_mse": pred_train_mse,
                "code_test_mse": code_test_mse,
                "code_train_mse": code_train_mse,
            }
            wandb.log(log_dic)

        else:
            wandb.log(
                {
                    "code_train_mse": code_train_mse,
                },
                step=step,
                commit=not step_show,
            )

        if code_train_mse < best_loss:
            best_loss = code_train_mse

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": code_test_mse,
                },
                f"{model_dir}/{run_name}.pt",
            )

    return code_test_mse


if __name__ == "__main__":
    main()
