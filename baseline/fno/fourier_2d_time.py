"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
from timeit import default_timer
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb
import torch
import numpy as np

from coral.utils.data.load_data import get_dynamics_data, set_seed
from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode
from fno.utilities3 import LpLoss, count_params
from fno.fno_model import FNO2d

################################################################
# configs
################################################################


@hydra.main(config_path="config/", config_name="fno.yaml")
def main(cfg: DictConfig) -> None:

    # Data is of the shape (number of samples, grid size)
    data_dir = cfg.data.dir
    dataset_name = cfg.data.dataset_name
    data_to_encode = cfg.data.data_to_encode
    same_grid = cfg.data.same_grid
    sub_tr = cfg.data.sub_tr
    sub_te = cfg.data.sub_te
    sub_from = cfg.data.sub_from
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    setting = cfg.data.setting
    sequence_length_in = cfg.data.sequence_length_in
    sequence_length_out = cfg.data.sequence_length_out
    sequence_length_ext = sequence_length_in+sequence_length_out
    sequence_length = None
    seed = cfg.data.seed

    modes = cfg.fno.modes
    width = cfg.fno.width

    batch_size = cfg.optim.batch_size
    learning_rate = cfg.optim.learning_rate
    epochs = cfg.optim.epochs
    iterations = epochs * (512 // batch_size)
    scheduler_step = cfg.optim.scheduler_step
    scheduler_gamma = cfg.optim.scheduler_gamma

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

    # RESULTS_DIR = "/data/serrano/functa2functa/fno/navier-stokes/"
    RESULTS_DIR = Path(os.getenv("WANDB_DIR")) / dataset_name / "fno"
    os.makedirs(str(RESULTS_DIR), exist_ok=True)

    set_seed(seed)

    ################################################################
    # load data and data normalization
    ################################################################
    u_train_ext = None
    u_test_ext = None
    u_train_out = None
    u_test_out = None
    grid_tr_ext = None
    grid_te_ext = None
    grid_tr_out = None
    grid_te_out = None

    (u_train, u_test, grid_tr, grid_te, u_train_out, u_test_out, grid_tr_out, grid_te_out, u_train_ext, u_test_ext, grid_tr_ext, grid_te_ext) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length,
        sub_from=sub_from,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
        setting=setting,
        sequence_length_in=sequence_length_in,
        sequence_length_out=sequence_length_out
    )

    print(
        f"data: {dataset_name}, u_tr: {u_train.shape}, u_test: {u_test.shape}"
    )
    print(
        f"grid: grid_tr: {grid_tr.shape}, grid_inp_te: {grid_te.shape}"
    )
    if u_train_ext is not None:
        print(
            f"data: {dataset_name}, u_t_ext: {u_train_ext.shape}, u_test_ext: {u_test_ext.shape}"
        )
        print(
            f"grid: grid_tr_ext: {grid_tr_ext.shape}, grid_inp_te_ext: {grid_te_ext.shape}"
        )

    if data_to_encode == None:
        run.tags = ("fno",) + \
            (dataset_name,) + (f"sub={sub_tr}",)
    else:
        run.tags = (
            ("fno",)
            + (dataset_name,)
            + (f"sub={sub_tr}",)
            + (data_to_encode,)
        )

    mean, sigma = u_train.mean(), u_train.std()
    # mean, sigma = (0, 1)

    # u_train = (u_train - mean) / sigma
    # u_test = (u_test - mean) / sigma
    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, 32, dataset_name, None
    )
    testset = TemporalDatasetWithCode(
        u_test, grid_te, 32, dataset_name, None
    )
    if u_train_out is not None:
        # u_train_out = (u_train_out - mean) / sigma
        # u_test_out = (u_test_out - mean) / sigma
        trainset_out = TemporalDatasetWithCode(
            u_train_out, grid_tr_out, 32, dataset_name, None
        )
        testset_out = TemporalDatasetWithCode(
            u_test_out, grid_te_out, 32, dataset_name, None
        )
    if u_train_ext is not None:
        # u_train_ext = (u_train_ext - mean) / sigma
        # u_test_ext = (u_test_ext - mean) / sigma
        trainset_ext = TemporalDatasetWithCode(
            u_train_ext, grid_tr_ext, 32, dataset_name, None
        )
        testset_ext = TemporalDatasetWithCode(
            u_test_ext, grid_te_ext, 32, dataset_name, None
        )

    ntrain = len(trainset)
    ntest = len(testset)

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    if u_train_ext is not None:
        train_loader_ext = torch.utils.data.DataLoader(
            trainset_ext,
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader_ext = torch.utils.data.DataLoader(
            testset_ext, batch_size=batch_size, shuffle=False
        )

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width).cuda()
    print(count_params(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    best_loss = np.inf

    for ep in range(epochs):
        step_show = ep % 100 == 0
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for u_batch, modulations, coord_batch, idx in train_loader:
            u_batch = u_batch.cuda()
            coord_batch = coord_batch.cuda()
            input_frame = u_batch[..., 0]
            batch_size = u_batch.shape[0]

            # print("u_batch.shape, coord_batch.shape, input_frame.shape : ",
            #       u_batch.shape, coord_batch.shape, input_frame.shape)

            loss = 0
            for t in range(sequence_length_in - 1):
                target_frame = u_batch[..., t+1]
                pred = model(input_frame)
                # print('device', pred.device, target_frame.device)
                # print('pred', pred.shape, target_frame.shape)
                loss += myloss(pred.view(batch_size, -1),
                               target_frame.view(batch_size, -1))
                optimizer.zero_grad()
                # loss = ((pred - target_frame)**2).mean()
                input_frame = pred.detach()
                xx = pred * sigma.cuda() + mean.cuda()
                yy = target_frame * sigma.cuda() + mean.cuda()
                train_l2 += ((xx.view(batch_size, -1) -
                              yy.view(batch_size, -1))**2).mean()*batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        test_l2 = 0.0
        test_l2_ext = 0.0
        train_l2_ext = 0.0
        with torch.no_grad():
            for u_batch, modulations, coord_batch, idx in test_loader:
                u_batch = u_batch.cuda()
                coord_batch = coord_batch.cuda()
                input_frame = u_batch[..., 0]
                batch_size = u_batch.shape[0]

                for t in range(sequence_length_in - 1):
                    target_frame = u_batch[..., t+1]
                    pred = model(input_frame)
                    loss = myloss(pred.view(batch_size, -1),
                                  target_frame.view(batch_size, -1))
                    # test_l2 += loss.item()
                    xx = pred * sigma.cuda() + mean.cuda()
                    yy = target_frame * sigma.cuda() + mean.cuda()
                    test_l2 += ((xx.view(batch_size, -1) -
                                yy.view(batch_size, -1))**2).mean()*batch_size

                    input_frame = pred.detach()

            if u_train_ext is not None:
                for u_batch, modulations, coord_batch, idx in train_loader_ext:
                    u_batch = u_batch.cuda()
                    coord_batch = coord_batch.cuda()
                    input_frame = u_batch[..., 0]
                    batch_size = u_batch.shape[0]

                    for t in range(sequence_length_ext - 1):
                        target_frame = u_batch[..., t+1]
                        pred = model(input_frame)
                        loss = myloss(pred.view(batch_size, -1),
                                      target_frame.view(batch_size, -1))
                        # test_l2 += loss.item()
                        xx = pred * sigma.cuda() + mean.cuda()
                        yy = target_frame * sigma.cuda() + mean.cuda()

                        # We don't use the first timesteps to compute the extrapolation loss
                        if t >= sequence_length_in:
                            train_l2_ext += ((xx.view(batch_size, -1) -
                                              yy.view(batch_size, -1))**2).mean()*batch_size

                        input_frame = pred.detach()

                for u_batch, modulations, coord_batch, idx in test_loader_ext:
                    u_batch = u_batch.cuda()
                    coord_batch = coord_batch.cuda()
                    input_frame = u_batch[..., 0]
                    batch_size = u_batch.shape[0]

                    for t in range(sequence_length_ext - 1):
                        target_frame = u_batch[..., t+1]
                        pred = model(input_frame)
                        loss = myloss(pred.view(batch_size, -1),
                                      target_frame.view(batch_size, -1))
                        # test_l2 += loss.item()
                        xx = pred * sigma.cuda() + mean.cuda()
                        yy = target_frame * sigma.cuda() + mean.cuda()

                        # We don't use the first timesteps to compute the extrapolation loss
                        if t >= sequence_length_in:
                            test_l2_ext += ((xx.view(batch_size, -1) -
                                            yy.view(batch_size, -1))**2).mean()*batch_size

                        input_frame = pred.detach()

            train_l2 /= (ntrain * sequence_length_in-1)
            train_l2_ext /= (ntrain * sequence_length_out-1)
            test_l2 /= (ntest * sequence_length_in-1)
            test_l2_ext /= (ntest * sequence_length_out-1)

            if step_show:
                if u_train_out is not None:
                    wandb.log(
                        {
                            "test_loss": test_l2,
                            "train_loss": train_l2,
                            'train_loss_ext': train_l2_ext,
                            'test_loss_ext': test_l2_ext,
                        }
                    )
                else:
                    wandb.log(
                        {
                            "test_loss": test_l2,
                            "train_loss": train_l2
                        },
                    )
            else:
                wandb.log(
                    {
                        "train_loss": train_l2,
                    },
                    step=ep,
                    commit=not step_show,
                )

            if train_l2 < best_loss:
                best_loss = train_l2

                torch.save(
                    {
                        "cfg": cfg,
                        "epoch": ep,
                        "fno": model.state_dict(),
                        "optimizer_inr": optimizer.state_dict(),
                        "loss": test_l2,
                        "grid_tr": grid_tr,
                        "grid_te": grid_te,
                        "mean_tr": mean,
                        "std_tr": sigma,
                    },
                    f"{RESULTS_DIR}/{run_name}.pt",
                )

        t2 = default_timer()
        print(ep, t2 - t1, train_l2, test_l2, train_l2_ext, test_l2_ext)

    result_dic = {'train_l2': train_l2, 'test_l2': test_l2,
                  'train_l2_ext': train_l2_ext, 'test_l2_ext': test_l2_ext}

    print(result_dic)

    return result_dic


if __name__ == "__main__":
    main()
