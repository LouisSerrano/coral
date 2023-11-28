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

@hydra.main(config_path="config/static/", config_name="different_inr.yaml")
def main(cfg: DictConfig) -> None:

    torch.set_default_dtype(torch.float32)

    # data
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    sub_tr = cfg.data.sub_tr
    seed = cfg.data.seed

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = (
        batch_size if cfg.optim.batch_size_val == None else cfg.optim.batch_size_val
    )
    lr_inr = cfg.optim.lr_inr
    gamma_step = cfg.optim.gamma_step
    lr_code = cfg.optim.lr_code
    meta_lr_code = cfg.optim.meta_lr_code
    weight_decay_code = cfg.optim.weight_decay_code
    inner_steps = cfg.optim.inner_steps
    test_inner_steps = cfg.optim.test_inner_steps
    epochs = cfg.optim.epochs
    lr_mlp = cfg.optim.lr_mlp
    weight_decay_mlp = cfg.optim.weight_decay_mlp

    # inr
    latent_dim = cfg.inr_in.latent_dim

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

    set_seed(seed)

    run.tags = (
        ("different-inr-regression",)
        + (dataset_name,)
        + (f"sub={sub_tr}",)
    )

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
        latent_dim=latent_dim,
        noise=0.02,
        task="static"
    )

    valset = DatasetClass(
        split="val",
        latent_dim=latent_dim,
        noise=0.02,
        task="static"
    )

    testset = DatasetClass(
        split="test",
        latent_dim=latent_dim,
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

    cfg.inr = cfg.inr_in
    inr_in = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )
    cfg.inr = cfg.inr_out
    inr_out = create_inr_instance(
        cfg, input_dim=input_dim, output_dim=output_dim, device="cuda"
    )

    alpha_in = nn.Parameter(torch.Tensor([lr_code]).cuda())
    alpha_out = nn.Parameter(torch.Tensor([lr_code]).cuda())

    optimizer_in = torch.optim.AdamW(
        [
            {"params": inr_in.parameters()},
            {"params": alpha_in, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    optimizer_out = torch.optim.AdamW(
        [
            {"params": inr_out.parameters()},
            {"params": alpha_out, "lr": meta_lr_code, "weight_decay": 0},
        ],
        lr=lr_inr,
        weight_decay=0,
    )

    best_loss = np.inf

    for step in range(epochs):
        fit_train_mse_in = 0
        fit_test_mse_in = 0
        fit_train_mse_out = 0
        fit_test_mse_out = 0
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
            inr_in.train()
            inr_out.train()

            graph.input_frame = graph.input[..., 2:] # discard the pos
            graph.output_frame = graph.images

            print('input', graph.input_frame.shape)
            print('output', graph.output_frame.shape)

            graph = graph.cuda()
            graph.images = graph.input_frame
            graph.modulations = torch.zeros_like(graph.z_vx[..., 0])
            graph.pos = graph.pos[..., 0]

            outputs = outer_step(
                inr_in,
                graph,
                inner_steps,
                alpha_in,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
            )

            optimizer_in.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_in.parameters(), clip_value=1.0)
            optimizer_in.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_in += loss.item() * n_samples
            z0 = outputs["modulations"].detach()

            graph.images = graph.output_frame
            graph.modulations = torch.zeros_like(graph.z_vx[..., 0])

            outputs = outer_step(
                inr_out,
                graph,
                inner_steps,
                alpha_out,
                is_train=True,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=use_rel_loss,
                loss_type="mse",
            )
            optimizer_out.zero_grad()
            outputs["loss"].backward(create_graph=False)
            nn.utils.clip_grad_value_(inr_out.parameters(), clip_value=1.0)
            optimizer_out.step()
            loss = outputs["loss"].cpu().detach()
            fit_train_mse_out += loss.item() * n_samples

            # mlp regression
            z1 = outputs["modulations"].detach()

        train_loss_in = fit_train_mse_in / (ntrain)
        train_loss_out = fit_train_mse_out / (ntrain)

        if use_pred_loss:
            for substep, (graph, idx) in enumerate(test_loader):  
                n_samples = len(graph)
                inr_in.train()
                inr_out.train()

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

            test_loss_in = fit_test_mse_in / (ntest)
            test_loss_out = fit_test_mse_out / (ntest)

        wandb.log(
            {
                "train_loss_in": train_loss_in,
                "test_loss_in": test_loss_in,
                "train_loss_out": train_loss_out,
                "test_loss_out": test_loss_out,
            },
            step=step,
        )

        if train_loss_out < best_loss:
            best_loss = train_loss_out
            # modulations_train = trainset.z.clone().detach()
            # modulations_test = valset.z.clone().detach()

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "inr_in": inr_in.state_dict(),
                    "inr_out": inr_out.state_dict(),
                    "optimizer_inr_in": optimizer_in.state_dict(),
                    "optimizer_inr_out": optimizer_out.state_dict(),
                    "loss": test_loss_out,
                    "alpha_in": alpha_in,
                    "alpha_out": alpha_out,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return test_loss_out


if __name__ == "__main__":
    main()
