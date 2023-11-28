import os
import sys
from pathlib import Path
from pickletools import OpcodeInfo

sys.path.append(str(Path(__file__).parents[1]))

from collections import OrderedDict

from coral.mlp import MLP
import einops
import h5py
import hydra
import numpy as np
import torch
import torch.nn as nn
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
from coral.utils.plot import show
from functools import partial
from coral.utils.data.graph_dataset import CylinderFlowDataset, AirfoilFlowDataset
from coral.utils.models.load_baseline import create_model_instance
import torch.utils.checkpoint as cp
from torch_geometric.nn import radius_graph
from coral.utils.data.load_data import set_seed

@hydra.main(config_path="../config/baselines", config_name="mlp.yaml")
def main(cfg: DictConfig) -> None:

    torch.set_default_dtype(torch.float32)

   # data
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    data_to_encode = cfg.data.data_to_encode
    sub_tr = cfg.data.sub_tr
    seed = cfg.data.seed

    # optim
    batch_size = cfg.optim.batch_size
    batch_size_val = batch_size
    lr = cfg.optim.lr
    weight_decay = cfg.optim.weight_decay
    gamma_step = cfg.optim.gamma_step
    epochs = cfg.optim.epochs

    #model
    model_type = cfg.model.model_type
    max_neighbours = cfg.model.max_neighbours
    radius = cfg.model.radius
   
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

    if dataset_name == "cylinder-flow":
        input_dim = 5
        output_dim = 3 
        DatasetClass = CylinderFlowDataset
    elif "airfoil-flow":
        input_dim = 6
        output_dim = 4
        DatasetClass = AirfoilFlowDataset
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding class.")

    latent_dim = 128

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

    run.tags = (
        ("baselines",) + (model_type,) + (dataset_name,)
    )

    ntrain = len(trainset)
    nval = len(valset)
    ntest = len(testset)

    train_loader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

    print("train", len(trainset))
    print("val", len(valset))

    # we map (pos, pressure, velocity)_{t=0} -> (pressure, velocity)_{t=T} 

    if dataset_name == "cylinder-flow":
        input_dim = 5
        output_dim = 3
    elif dataset_name == "airfoil-flow":
        input_dim = 6
        output_dim = 4
    else:
        raise NotImplementedError(f"The dataset ${dataset_name} does not have a corresponding input/output dims.")
    
    
    print('input_dim', input_dim, output_dim)
    model = create_model_instance(cfg, input_dim=input_dim, output_dim=output_dim)
    model = model.cuda()

    print('model', model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=gamma_step,
        patience=500,
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
        velocity_train_mse = 0
        velocity_test_mse = 0
        pressure_train_mse = 0
        pressure_test_mse = 0
        density_train_mse = 0
        density_test_mse = 0

        use_rel_loss = step % 10 == 0
        step_show = step % 100 == 0

        for substep, (graph, idx) in enumerate(train_loader):
            model.train()
            n_samples = len(graph)

            #graph.input = graph.input.to(torch.float64)

            # p of shape (N, 1, T)
            # v of shape (N, 2, T)
            
            #graph.input = torch.cat([graph.pos[..., 0], graph.p[..., 0], graph.v[..., 0]], axis=-1)
            #graph.images = torch.cat([graph.p[..., 1], graph.v[..., 1]], axis=-1)

            if model_type in ['sage', 'gunet', 'mppde']:
                graph.edge_index = radius_graph(
                                        graph.pos[..., 0],
                                        radius,
                                        graph.batch,
                                        loop=False,
                                        max_num_neighbors=max_neighbours,)
                #print("graph", graph.edge_index.dtype)
            graph = graph.cuda()
            #print("graph", graph.input.dtype, graph.input.shape)

            if model_type == "mlp":
                pred = model(graph.input)
            elif model_type == "sage":
                pred = model(graph.input, graph.edge_index)
            elif model_type == "gunet":
                pred = model(graph.input, graph.edge_index, graph.batch)
            elif model_type == "mppde":
                pred = model(graph)

            loss = ((pred - graph.images)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_train_mse += loss.item() * n_samples
            if dataset_name == "cylinder-flow":
                pressure_train_mse += ((pred[..., :1] - graph.images[..., :1])**2).mean().item() * n_samples
                velocity_train_mse += ((pred[..., 1:] - graph.images[..., 1:])**2).mean().item() * n_samples
            elif dataset_name == "airfoil-flow":
                pressure_train_mse += ((pred[..., :1] - graph.images[..., :1])**2).mean().item() * n_samples
                density_train_mse += ((pred[..., 1:2] - graph.images[..., 1:2])**2).mean().item() * n_samples
                velocity_train_mse += ((pred[..., 2:] - graph.images[..., 2:])**2).mean().item() * n_samples

        pred_train_mse = pred_train_mse / ntrain
        velocity_train_mse = velocity_train_mse / ntrain
        pressure_train_mse = pressure_train_mse / ntrain

        scheduler.step(pred_train_mse)

        if step_show:
            for graph, idx in test_loader:
                model.eval()
                n_samples = len(graph)

                # p of shape (N, 1, T)
                # v of shape (N, 2, T)

                #graph.input = graph.input.to(torch.float64)

                #graph.input = torch.cat([graph.pos[..., 0], graph.p[..., 0], graph.v[..., 0]], axis=-1)
                #graph.images = torch.cat([graph.p[..., 1], graph.v[..., 1]], axis=-1)

                if model_type in ['sage', 'gunet', 'mppde']:
                    graph.edge_index = radius_graph(
                                        graph.pos[..., 0],
                                        radius,
                                        graph.batch,
                                        loop=False,
                                        max_num_neighbors=max_neighbours,)
                graph = graph.cuda()

                with torch.no_grad():
                    if model_type == "mlp":
                        pred = model(graph.input)
                    elif model_type == "sage":
                        pred = model(graph.input, graph.edge_index)
                    elif model_type == "gunet":
                        pred = model(graph.input, graph.edge_index, graph.batch)
                    elif model_type == "mppde":
                        pred = model(graph)
                
                #print('pred', pred.shape, pred.mean(), pred.std())

                loss = ((pred - graph.images)**2).mean()

                pred_test_mse += loss.item() * n_samples
                if dataset_name == "cylinder-flow":
                    pressure_test_mse += ((pred[..., :1] - graph.images[..., :1])**2).mean().item() * n_samples
                    velocity_test_mse += ((pred[..., 1:] - graph.images[..., 1:])**2).mean().item() * n_samples
                elif dataset_name == "airfoil-flow":
                    pressure_test_mse += ((pred[..., :1] - graph.images[..., :1])**2).mean().item() * n_samples
                    density_test_mse += ((pred[..., 1:2] - graph.images[..., 1:2])**2).mean().item() * n_samples
                    velocity_test_mse += ((pred[..., 2:] - graph.images[..., 2:])**2).mean().item() * n_samples

            pred_test_mse = pred_test_mse / ntest
            velocity_test_mse = velocity_test_mse / ntest
            pressure_test_mse = pressure_test_mse / ntest

        if step_show:
            log_dic = {"pred_train_mse": pred_train_mse,
                "pred_test_mse": pred_test_mse,
                "velocity_train_mse": velocity_train_mse,
                "veloctiy_test_mse": velocity_test_mse,
                "pressure_train_mse": pressure_train_mse,
                "pressure_test_mse": pressure_test_mse,
            }
            if dataset_name == 'airfoil-flow':
                log_dic.update({"density_train_mse": density_train_mse,
                                "density_test_mse": density_test_mse})
            wandb.log(log_dic)

        else:
            log_dic= {"pred_train_mse": pred_train_mse,
                    "velocity_train_mse": velocity_train_mse,
                    "pressure_train_mse": pressure_train_mse,}
            
            if dataset_name == 'airfoil-flow':
                log_dic.update({"density_train_mse": density_train_mse})
    
    
            wandb.log(log_dic,
                step=step,
                commit=not step_show,
            )

        if pred_train_mse < best_loss:
            best_loss = pred_train_mse

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": pred_test_mse,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    return pred_test_mse


if __name__ == "__main__":
    main()
