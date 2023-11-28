import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo

sys.path.append(str(Path(__file__).parents[1]))

import einops
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torchdiffeq import odeint

from coral.losses import batch_mse_rel_fn
from coral.mfn import FourierNet, HyperMAGNET, HyperMultiscaleBACON
from coral.mlp import MLP, Derivative, ResNet
from coral.siren import ModulatedSiren
from coral.utils.data.load_data import set_seed
from coral.utils.plot import show
from coral.utils.data.graph_dataset import CylinderFlowDataset, AirfoilFlowDataset
from coral.utils.models.load_inr import create_inr_instance
import torch.utils.checkpoint as cp
from coral.metalearning import graph_outer_step as outer_step
from torch_geometric.loader import DataLoader

@hydra.main(config_path="../config/static/", config_name="regression.yaml")
def main(cfg: DictConfig) -> None:

    torch.set_default_dtype(torch.float32)

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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

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

    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "inr"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)
    wandb.log({"results_dir": str(RESULTS_DIR)}, step=0, commit=False)

    set_seed(seed)

    run.tags = (
        ("static-regression",)
        + (dataset_name,)
    )


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


    if dataset_name == "cylinder-flow":
         # trainset coords of shape (N, L, d_in, 2)
        input_dim = 2
        output_dim = 3
        DatasetClass = CylinderFlowDataset
    elif "airfoil-flow":
        DatasetClass = AirfoilFlowDataset
        input_dim = 2
        output_dim = 4
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    trainset = DatasetClass(
        split="train",
        latent_dim=latent_dim_in,
        noise=0.02,
        task="static"
    )

    valset = DatasetClass(
        split="val",
        latent_dim=latent_dim_in,
        noise=0.02,
        task="static"
    )

    testset = DatasetClass(
        split="test",
        latent_dim=latent_dim_in,
        noise=0.02,
        task="static"
    )

    ntrain = len(trainset)
    nval = len(valset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    print("train", len(trainset))
    print("val", len(valset))
   
    # load inr weights
    load_cfg.inr = load_cfg.inr_in
    inr_in = create_inr_instance(
        load_cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    inr_in.load_state_dict(input_inr["inr_in"])
    inr_in.eval()
    alpha_in = input_inr['alpha_in']

    
    load_cfg.inr = load_cfg.inr_out
    inr_out = create_inr_instance(
        load_cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    inr_out.load_state_dict(input_inr["inr_out"])
    inr_out.eval()
    alpha_out = input_inr['alpha_out']


    model = ResNet(
        input_dim=load_cfg.inr_in.latent_dim,
        hidden_dim=cfg.model.width,
        output_dim=load_cfg.inr_out.latent_dim,
        depth=cfg.model.depth,
        dropout=cfg.model.dropout,
        activation=cfg.model.activation,
    ).cuda()

    optimizer_pred = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )  # 1e-4

    # 1. get modulations

    assert latent_dim_in == latent_dim_out

    step = 0
    z_train = torch.zeros(ntrain, latent_dim_in, 2)
    z_test = torch.zeros(ntest, latent_dim_in, 2)

    fit_train_mse_in = 0
    fit_test_mse_in = 0
    fit_train_mse_out = 0
    fit_test_mse_out = 0
    use_rel_loss = step % 10 == 0
    use_pred_loss = step % 20 == 0
    step_show = step % 50 == 0
    for substep, (graph, idx) in enumerate(
        train_loader
    ):  
        n_samples = len(graph)
        inr_in.train()
        inr_out.train()
        model.train()
        
        mask = ... #torch.randperm(graph.input.shape[0])[n_samples*1000]
        
        graph.input_frame = graph.input[mask, 2:] # discard the pos
        graph.output_frame = graph.images

        graph = graph.cuda()
        graph.images = graph.input_frame
        graph.modulations = torch.zeros_like(graph.z_vx[..., 0])
        graph.pos = graph.pos[..., 0]

        outputs = outer_step(
            inr_in,
            graph,
            inner_steps,
            alpha_in,
            is_train=False,
            return_reconstructions=False,
            gradient_checkpointing=False,
            use_rel_loss=use_rel_loss,
            loss_type="mse",
        )

        loss = outputs["loss"].cpu().detach()
        fit_train_mse_in += loss.item() * n_samples
        z0 = outputs["modulations"].detach()
        z_train[idx, :, 0] = z0.cpu()

        graph.images = graph.output_frame
        graph.modulations = torch.zeros_like(graph.z_vx[..., 0])

        outputs = outer_step(
            inr_out,
            graph,
            inner_steps,
            alpha_out,
            is_train=False,
            return_reconstructions=False,
            gradient_checkpointing=False,
            use_rel_loss=use_rel_loss,
            loss_type="mse",
        )
        loss = outputs["loss"].cpu().detach()
        fit_train_mse_out += loss.item() * n_samples

        # mlp regression
        z1 = outputs["modulations"].detach()
        z_train[idx, :, 1] = z1.cpu()

    train_loss_in = fit_train_mse_in / (ntrain)
    train_loss_out = fit_train_mse_out / (ntrain)

    for substep, (graph, idx) in enumerate(test_loader):  
        n_samples = len(graph)
        inr_in.train()
        inr_out.train()
        model.train()

        graph.input_frame = graph.input[..., 2:] # discard the pos
        graph.output_frame = graph.images

        graph = graph.cuda()
        graph.images = graph.input_frame
        graph.modulations = torch.zeros_like(graph.z_vx[..., 0])
        graph.pos = graph.pos[..., 0]

        outputs = outer_step(
            inr_in,
            graph,
            inner_steps,
            alpha_in,
            is_train=False,
            return_reconstructions=False,
            gradient_checkpointing=False,
            use_rel_loss=use_rel_loss,
            loss_type="mse",
        )
        loss = outputs["loss"].cpu().detach()
        fit_test_mse_in += loss.item() * n_samples
        z0 = outputs["modulations"].detach()
        
        z_test[idx, :, 0] = z0.cpu()

        graph.images = graph.output_frame
        graph.modulations = torch.zeros_like(graph.z_vx[..., 0])

        outputs = outer_step(
            inr_out,
            graph,
            inner_steps,
            alpha_out,
            is_train=False,
            return_reconstructions=False,
            gradient_checkpointing=False,
            use_rel_loss=use_rel_loss,
            loss_type="mse",
        )

        loss = outputs["loss"].cpu().detach()
        fit_test_mse_out += loss.item() * n_samples

        # mlp regression
        z1 = outputs["modulations"].detach()
        z_test[idx, :, 1] = z1.cpu()


    test_loss_in = fit_test_mse_in / (ntest)
    test_loss_out = fit_test_mse_out / (ntest)

    print('train in', train_loss_in, 'train_out', train_loss_out)
    print('test in', test_loss_in, 'test_out', test_loss_out)

    if dataset_name == 'airfoil-flow':
        mu = z_train[..., 0].mean()
        std = z_train[..., 0].std()

        z_train[..., 0] = (z_train[..., 0] - mu)/std
        z_test[..., 0] = (z_test[..., 0] - mu)/std

        mu_u = z_train[..., 1].mean() 
        std_u = z_train[..., 1].std() 
        z_train[..., 1] = (z_train[..., 1] - mu_u)/std_u
        z_test[..., 1] = (z_test[..., 1] - mu_u)/std_u

    elif dataset_name == 'cylinder-flow':
        mu = z_train[..., 0].mean(0)
        std = z_train[..., 0].std(0)

        z_train[..., 0] = (z_train[..., 0] - mu)/std
        z_test[..., 0] = (z_test[..., 0] - mu)/std

        mu_u = z_train[..., 1].mean(0) 
        std_u = z_train[..., 1].std(0) 
        z_train[..., 1] = (z_train[..., 1] - mu_u)/std_u
        z_test[..., 1] = (z_test[..., 1] - mu_u)/std_u


    # 2. Start training the inference model

    best_loss = np.inf

    for step in range(epochs):
        pred_train_mse = 0
        pred_test_mse = 0
        code_train_mse = 0
        code_test_mse = 0
        use_rel_loss = step % 10 == 0
        use_pred_loss = step % 20 == 0
        step_show = step % 50 == 0
        for substep, (graph, idx) in enumerate(
            train_loader
        ):  
            n_samples = len(graph)
            model.train()

            graph.input_frame = graph.input[..., 2:] # discard the pos
            graph.output_frame = graph.images
            graph.pos = graph.pos[..., 0]
            graph = graph.cuda()

            z_input = z_train[idx, :, 0].cuda()
            z_output = z_train[idx, : , 1].cuda()
                
            z_pred = model(z_input)
            loss = ((z_pred - z_output) ** 2).mean()
            
            optimizer_pred.zero_grad()
            loss.backward()
            optimizer_pred.step()
            code_train_mse += loss.item() * n_samples

            if use_pred_loss:
                with torch.no_grad():
                    z_ = z_pred * std_u.cuda() + mu_u.cuda()
                    u_pred = inr_out.modulated_forward(graph.pos, z_[graph.batch])
                pred_train_mse += ((u_pred - graph.output_frame)**2).mean() * n_samples

        code_train_loss = code_train_mse / ntrain

        if use_pred_loss:
            pred_train_loss = pred_train_mse / ntrain
            print(step, 'train: code', code_train_loss, 'pred', pred_train_loss)

        if use_pred_loss:
            for substep, (graph, idx) in enumerate(test_loader):  
                n_samples = len(graph)
                model.eval()

                graph.input_frame = graph.input[..., 2:] # discard the pos
                graph.output_frame = graph.images
                graph.pos = graph.pos[..., 0]
                graph = graph.cuda()
                
                z_input = z_test[idx, :, 0].cuda()
                z_output = z_test[idx, : , 1].cuda()

                z_pred = model(z_input)
                loss = ((z_pred - z_output) ** 2).mean()
                
                code_test_mse += loss.item() * n_samples

                if use_pred_loss:
                    with torch.no_grad():
                        z_ = z_pred * std_u.cuda() + mu_u.cuda()
                        u_pred = inr_out.modulated_forward(graph.pos, z_[graph.batch])
                    pred_test_mse += ((u_pred - graph.output_frame)**2).mean() * n_samples

            code_test_loss = code_test_mse / ntest        
            pred_test_loss = pred_test_mse / (ntest)
            print(step, 'test: code', code_test_loss, 'pred', pred_test_loss)


        if use_pred_loss:
            COMMIT = not use_rel_loss
            wandb.log(
                {
                    "code_test_loss": code_test_loss,
                    "code_train_loss": code_train_loss,
                    "pred_test_loss": pred_test_loss,
                    "pred_train_loss": pred_train_loss,
                },
                step=step,
                commit=COMMIT,
            )

            if use_pred_loss:
                COMMIT = True
                wandb.log(
                    {"train_pred_loss": pred_train_loss, "test_pred_loss": pred_test_loss},
                    step=step,
                    commit=COMMIT,
                )

            if code_train_mse < best_loss:
                best_loss = code_train_mse
                # modulations_train = trainset.z.clone().detach()
                # modulations_test = valset.z.clone().detach()

                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer_pred.state_dict(),
                        "loss": test_loss_out,
                        "alpha_in": alpha_in,
                        "alpha_out": alpha_out,
                        "mu": mu,
                        "mu_u": mu_u,
                        "sigma": std,
                        "sigma_u": std_u
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )

    return code_test_loss


if __name__ == "__main__":
    main()
