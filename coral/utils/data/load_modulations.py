import os
from pathlib import Path

import einops
import torch
from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader as GeometricLoader
from coral.metalearning import outer_step, graph_outer_step


def load_operator_modulations(
    trainset,
    valset,
    inr_a,
    inr_u,
    run_dir,
    run_name,
    inner_steps=3,
    alpha_a=0.01,
    alpha_u=0.01,
    batch_size=4,
    data_to_encode=None,
    try_reload=False,
):
    run_dir = Path(run_dir)

    if try_reload:
        try:
            if data_to_encode is None:
                return torch.load(run_dir / f"{run_name}.pt")
            else:
                return torch.load(run_dir / f"{data_to_encode}/{run_name}.pt")
        except:
            try_reload = False
    else:
        train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        ntrain = len(trainset)
        ntest = len(valset)
        latent_dim_a = trainset.latent_dim_a
        latent_dim_u = trainset.latent_dim_u

        fit_a_train_mse = 0
        fit_u_train_mse = 0
        fit_a_test_mse = 0
        fit_u_test_mse = 0

        za_tr = torch.zeros(ntrain, latent_dim_a)
        zu_tr = torch.zeros(ntrain, latent_dim_u)
        za_te = torch.zeros(ntest, latent_dim_a)
        zu_te = torch.zeros(ntest, latent_dim_u)

        inr_a.eval()
        inr_u.eval()

        for substep, (
            batch_a,
            batch_u,
            batch_za,
            batch_zu,
            batch_coord,
            idx,
        ) in enumerate(train_loader):
            batch_a = batch_a.cuda()
            batch_u = batch_u.cuda()
            batch_za = batch_za.cuda()
            batch_zu = batch_zu.cuda()
            batch_coord = batch_coord.cuda()
            n_samples = batch_a.shape[0]

            outputs = outer_step(
                inr_a,
                batch_coord,
                batch_a,
                inner_steps,
                alpha_a,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
                loss_type="mse",
                modulations=torch.zeros_like(batch_za),
            )
            za_tr[idx] = outputs["modulations"].cpu().detach()
            loss = outputs["loss"]
            fit_a_train_mse += loss.item() * n_samples

            outputs = outer_step(
                inr_u,
                batch_coord,
                batch_u,
                inner_steps,
                alpha_u,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
                loss_type="mse",
                modulations=torch.zeros_like(batch_zu),
            )
            zu_tr[idx] = outputs["modulations"].cpu().detach()
            loss = outputs["loss"]
            fit_u_train_mse += loss.item() * n_samples

        print(
            f"Train, input loss: {fit_a_train_mse / ntrain}, output loss: {fit_u_train_mse / ntrain}"
        )

        for substep, (
            batch_a,
            batch_u,
            batch_za,
            batch_zu,
            batch_coord,
            idx,
        ) in enumerate(val_loader):
            batch_a = batch_a.cuda()
            batch_u = batch_u.cuda()
            batch_za = batch_za.cuda()
            batch_zu = batch_zu.cuda()
            batch_coord = batch_coord.cuda()
            n_samples = batch_a.shape[0]

            outputs = outer_step(
                inr_a,
                batch_coord,
                batch_a,
                inner_steps,
                alpha_a,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
                loss_type="mse",
                modulations=torch.zeros_like(batch_za),
            )
            za_te[idx] = outputs["modulations"].cpu().detach()
            loss = outputs["loss"]
            fit_a_test_mse += loss.item() * n_samples

            outputs = outer_step(
                inr_u,
                batch_coord,
                batch_u,
                inner_steps,
                alpha_u,
                is_train=False,
                return_reconstructions=True,
                gradient_checkpointing=False,
                use_rel_loss=False,
                loss_type="mse",
                modulations=torch.zeros_like(batch_zu),
            )
            zu_te[idx] = outputs["modulations"].cpu().detach()
            loss = outputs["loss"]
            fit_u_test_mse += loss.item() * n_samples

        print(
            f"Test, input loss: {fit_a_test_mse / ntest}, output loss: {fit_u_test_mse / ntest}"
        )

        modulations = {
            "za_train": za_tr,
            "za_test": za_te,
            "zu_train": zu_tr,
            "zu_test": zu_te,
            "fit_test_input_loss": fit_a_test_mse / ntest,
            "fit_test_output_loss": fit_u_test_mse / ntest,
            "fit_train_input_loss": fit_a_train_mse / ntrain,
            "fit_train_output_loss": fit_u_train_mse / ntrain,
        }

        if data_to_encode is None:
            pass
            #os.makedirs(str(run_dir / f"/{run_name}.pt"), exist_ok=True)
            #torch.save(modulations, str(run_dir / f"/{run_name}.pt"))
        else:
            os.makedirs(
                str(run_dir / f"{data_to_encode}/{run_name}.pt"), exist_ok=True)
            torch.save(modulations, str(
                run_dir / f"/{data_to_encode}/{run_name}.pt"))

        return modulations

def eval_modulation_loop(loader, inr, inner_steps, alpha, z, T):
    fit_mse = 0
    for substep, (batch_v, batch_z, batch_coord, idx) in enumerate(loader):
        batch_v = batch_v.cuda()
        batch_z = batch_z.cuda()
        batch_coord = batch_coord.cuda()
        n_samples = batch_v.shape[0]

        if batch_coord.shape[-2] == 2:
            batch_v = einops.rearrange(batch_v, 'b ... t -> (b t) ...')#.unsqueeze(-1)
        else:
            batch_v = einops.rearrange(batch_v, 'b ... t -> (b t) ...').unsqueeze(-1)

        batch_z = einops.rearrange(batch_z, 'b ... t -> (b t) ...')
        batch_coord = einops.rearrange(batch_coord, 'b ... t -> (b t) ...')

        outputs = outer_step(
            inr,
            batch_coord,
            batch_v,
            inner_steps,
            alpha,
            is_train=False,
            return_reconstructions=True,
            gradient_checkpointing=False,
            use_rel_loss=False,
            loss_type="mse",
            modulations=torch.zeros_like(batch_z),
        )
        tmp_mod = outputs["modulations"].cpu().detach()
        tmp_mod = einops.rearrange(tmp_mod, "(b t) ... -> b ... t", t=T)
        z[idx] = tmp_mod
        loss = outputs["loss"]
        fit_mse += loss.item() * n_samples
    return z, fit_mse

def load_dynamics_modulations(
    trainset,
    trainset_extra,
    valset,
    inr,
    run_dir,
    run_name,
    inner_steps=3,
    alpha=0.01,
    batch_size=2,
    data_to_encode=None,
    try_reload=False,
):
    """WARNING : This function assumes that we can encode a full trajectory"""
    run_dir = Path(run_dir)
    if try_reload:
        try:
            return torch.load(run_dir / f"{run_name}.pt")
        except:
            try_reload = False
    else:
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        train_extra_loader = DataLoader(trainset_extra, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)

        ntrain = len(trainset)
        ntest = len(valset)
        latent_dim = trainset.latent_dim
        T_train = trainset.T
        T_test = valset.T

        # add reconstruction loss as in FNO paper
        z_tr = torch.zeros(ntrain, latent_dim, T_train)
        z_tr_extra = torch.zeros(ntrain, latent_dim, T_test)
        z_te = torch.zeros(ntest, latent_dim, T_test)

        inr.eval()

        z_tr, fit_train_mse = eval_modulation_loop(train_loader, inr, inner_steps, alpha, z_tr, T_train)
        print(f"Train, average loss: {fit_train_mse / ntrain}")

        z_tr_extra, fit_train_extra_mse = eval_modulation_loop(train_extra_loader, inr, inner_steps, alpha, z_tr_extra, T_test)

        print(f"Train extra, average loss: {fit_train_extra_mse / ntrain}")

        z_te, fit_test_mse = eval_modulation_loop(test_loader, inr, inner_steps, alpha, z_te, T_test)

        print(f"Test, average loss: {fit_test_mse / ntest}")

        modulations = {
            "z_train": z_tr,
            "z_train_extra": z_tr_extra,
            "z_test": z_te,
            "fit_train_mse": fit_train_mse / ntrain,
            "fit_train_extra_mse": fit_train_extra_mse / ntrain,
            "fit_test_mse": fit_test_mse / ntest,
        }

        os.makedirs(str(run_dir), exist_ok=True)
        torch.save(modulations, str(run_dir / f"{run_name}.pt"))

        return modulations


def load_graph_modulations(
    trainset,
    valset,
    inr,
    run_dir,
    run_name,
    data_to_encode="velocity",
    inner_steps=3,
    alpha=0.01,
    batch_size=8,
    try_reload=False,
):
    """WARNING : This function assumes that we can encode a full trajectory"""
    run_dir = Path(run_dir)
    if try_reload:
        try:
            return torch.load(run_dir / f"{run_name}.pt")
        except:
            try_reload = False
    else:
        train_loader = GeometricLoader(trainset, batch_size=batch_size, shuffle=False)
        val_loader = GeometricLoader(valset, batch_size=batch_size, shuffle=False)

        ntrain = len(trainset)
        ntest = len(valset)
        latent_dim = trainset.latent_dim
        T = trainset.T

        fit_train_mse = 0
        fit_test_mse = 0
        # add reconstruction loss as in FNO paper

        z_tr = torch.zeros(ntrain, latent_dim, T)
        z_te = torch.zeros(ntest, latent_dim, T)

        inr.eval()

        for substep, (graph, idx) in enumerate(train_loader):
            n_samples = len(graph)

            if data_to_encode == "vx":
                graph.images = graph.v[:, 0, :].unsqueeze(1)
                graph.modulations = graph.z_vx

            elif data_to_encode == "vy":
                graph.images = graph.v[:, 1, :].unsqueeze(1)
                graph.modulations = graph.z_vy

            elif data_to_encode == "velocity":
                graph.images = graph.v
                graph.modulations = graph.z_v

            elif data_to_encode == "pressure":
                graph.images = graph.p
                graph.modulations = graph.z_p
            
            elif data_to_encode == "density":
                graph.images = graph.rho
                graph.modulations = graph.z_rho

            elif data_to_encode == "density":
                graph.images = graph.rho
                graph.modulations = graph.z_rho

            elif data_to_encode == "geometry":
                # we only need one sample for the geometry
                graph.images = graph.nodes[..., 0]
                graph.modulations = graph.z_geo[..., 0]
                graph.pos = graph.pos[..., 0]

            if data_to_encode != "geometry":
                graph.images = torch.cat(
                    [graph.images[:, :, 0], graph.images[:, :, 1]], axis=0
                )
                graph.modulations = torch.cat(
                    [graph.modulations[:, :, 0], graph.modulations[:, :, 1]], axis=0
                )
                graph.batch = torch.cat(
                    [graph.batch, graph.batch + n_samples], axis=0
                ).cuda()
                graph.pos = torch.cat([graph.pos[:, :, 0], graph.pos[:, :, 1]], axis=0)

                # old
                # graph.images = einops.rearrange(graph.images, "b ... t -> (b t) ...")
                # graph.modulations = einops.rearrange(graph.modulations, "b ... t -> (b t) ...")
                # graph.pos = einops.rearrange(graph.pos, "b ... t -> (b t) ...")
                # this puts in the batch size (v^0_0, v^0_1, v^1_0, v^1_1,)
                # graph.batch = graph.batch.repeat_interleave(2)
                # graph.batch[1::2] += n_samples # we need to treat differently the input and output fields

            # print('modultions', graph.modulations.shape)

            graph = graph.cuda()

            outputs = graph_outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                is_train=False,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=True,
                loss_type="mse",
            )

            tmp_mod = outputs["modulations"].cpu().detach()
            if data_to_encode != "geometry":
                # old
                # tmp_mod = einops.rearrange(tmp_mod, "(b t) ... -> b ... t", t=T)
                tmp_mod = einops.rearrange(tmp_mod, "(t b) l -> b l t", t=2)
            z_tr[idx] = tmp_mod
            loss = outputs["loss"]
            fit_train_mse += loss.item() * n_samples

        print(f"Train, average loss: {fit_train_mse / ntrain}")

        for substep, (graph, idx) in enumerate(val_loader):
            n_samples = len(graph)

            if data_to_encode == "vx":
                graph.images = graph.v[:, 0, :].unsqueeze(1)
                graph.modulations = graph.z_vx

            elif data_to_encode == "vy":
                graph.images = graph.v[:, 1, :].unsqueeze(1)
                graph.modulations = graph.z_vy

            elif data_to_encode == "velocity":
                graph.images = graph.v
                graph.modulations = graph.z_v

            elif data_to_encode == "pressure":
                graph.images = graph.p
                graph.modulations = graph.z_p
            
            elif data_to_encode == "density":
                graph.images = graph.rho
                graph.modulations = graph.z_rho

            elif data_to_encode == "density":
                graph.images = graph.rho
                graph.modulations = graph.z_rho

            elif data_to_encode == "geometry":
                # we only need one sample for the geometry
                graph.images = graph.nodes[..., 0]
                graph.modulations = graph.z_geo[..., 0]
                graph.pos = graph.pos[..., 0]

            if data_to_encode != "geometry":
                graph.images = torch.cat(
                    [graph.images[:, :, 0], graph.images[:, :, 1]], axis=0
                )
                graph.modulations = torch.cat(
                    [graph.modulations[:, :, 0], graph.modulations[:, :, 1]], axis=0
                )
                graph.batch = torch.cat(
                    [graph.batch, graph.batch + n_samples], axis=0
                ).cuda()
                graph.pos = torch.cat([graph.pos[:, :, 0], graph.pos[:, :, 1]], axis=0)
                # graph.images = einops.rearrange(graph.images, "b ... t -> (b t) ...")
                # graph.modulations = einops.rearrange(graph.modulations, "b ... t -> (b t) ...")
                # graph.pos = einops.rearrange(graph.pos, "b ... t -> (b t) ...")
                # this puts in the batch size (v^0_0, v^0_1, v^1_0, v^1_1,)
                # graph.batch = graph.batch.repeat_interleave(2)
                # graph.batch[1::2] += n_samples # we need to treat differently the input and output fields

            graph = graph.cuda()

            outputs = graph_outer_step(
                inr,
                graph,
                inner_steps,
                alpha,
                is_train=False,
                return_reconstructions=False,
                gradient_checkpointing=False,
                use_rel_loss=True,
                loss_type="mse",
            )
            tmp_mod = outputs["modulations"].cpu().detach()
            if data_to_encode != "geometry":
                # tmp_mod = einops.rearrange(tmp_mod, "(b t) ... -> b ... t", t=T)
                tmp_mod = einops.rearrange(tmp_mod, "(t b) l -> b l t", t=2)
            z_te[idx] = tmp_mod
            loss = outputs["loss"]
            fit_test_mse += loss.item() * n_samples

        print(f"Test, average loss: {fit_test_mse / ntest}")

        modulations = {
            "z_train": z_tr,
            "z_test": z_te,
            "fit_train_mse": fit_train_mse / ntrain,
            "fit_test_mse": fit_test_mse / ntest,
        }

        os.makedirs(str(run_dir), exist_ok=True)
        torch.save(modulations, str(run_dir / f"{run_name}.pt"))

        return modulations
