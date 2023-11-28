"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
from coral.utils.data.load_data import get_dynamics_data, set_seed
from dynamics_dataset import TemporalDatasetWithCode
from utilities3 import *
import torch.nn.functional as F
from timeit import default_timer
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import wandb

sys.path.append(str(Path(__file__).parents[1]))

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(
                x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, num_channels):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """
        self.num_channels = num_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(2 + num_channels, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(
            self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat(
            [batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat(
            [batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


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
    ntrain = cfg.data.ntrain
    ntest = cfg.data.ntest
    seq_inter_len = cfg.data.seq_inter_len
    seq_extra_len = cfg.data.seq_extra_len

    seed = cfg.data.seed

    modes = cfg.fno.modes
    width = cfg.fno.width

    batch_size = cfg.optim.batch_size
    learning_rate = cfg.optim.learning_rate
    epochs = cfg.optim.epochs
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

    (u_train, u_eval_extrapolation, u_test, grid_tr, grid_tr_extra, grid_te) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        seq_inter_len=seq_inter_len,
        seq_extra_len=seq_extra_len,
        sub_tr=sub_tr,
        sub_te=sub_te,
        same_grid=same_grid,
    )

    print(f"data: {dataset_name}, u_train: {u_train.shape}, u_train_eval: {u_eval_extrapolation.shape}, u_test: {u_test.shape}")
    print(f"grid: grid_tr: {grid_tr.shape}, grid_tr_extra: {grid_tr_extra.shape}, grid_te: {grid_te.shape}")

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

    trainset = TemporalDatasetWithCode(
        u_train, grid_tr, 0, dataset_name, data_to_encode
    )

    trainset_out = TemporalDatasetWithCode(
        u_eval_extrapolation, grid_tr_extra, 0, dataset_name, data_to_encode
    )
    
    testset = TemporalDatasetWithCode(
        u_test, grid_te, 0, dataset_name, data_to_encode
    )

    mean, sigma = u_train.mean(), u_train.std()
    u_train = (u_train - mean) / sigma
    u_test = (u_test - mean) / sigma
    
    ntrain = u_train.shape[0]
    ntest = u_test.shape[0]
    num_channels = trainset[0][0].shape[-2]

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True, # TODO : here shuffle to False because error cuda (?!)
        num_workers=1,
        pin_memory=True,
    )

    valid_loader = torch.utils.data.DataLoader(
        trainset_out,
        batch_size=batch_size,
        shuffle=True, # TODO : here shuffle to False because error cuda (?!)
        num_workers=1,
        pin_memory=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=True, # TODO : here shuffle to False because error cuda (?!)
        num_workers=1,
    )
    

    ################################################################
    # training and evaluation
    ################################################################
    model = FNO2d(modes, modes, width, num_channels).cuda()
    print(count_params(model))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    #myloss = LpLoss(size_average=False)
    myloss = nn.MSELoss()
    best_loss = np.inf

    for ep in range(epochs):
        step_show = ep % 100 == 0
        model.train()
        t1 = default_timer()
        pred_train_mse = 0
        for u_batch, modulations, coord_batch, idx in train_loader:
            u_batch = u_batch.cuda()
            coord_batch = coord_batch.cuda()
            input_frame = u_batch[..., 0]
            batch_size = u_batch.shape[0]

            loss = 0
            for t in range(seq_inter_len - 1):
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
                pred_train_mse += ((xx.view(batch_size, -1) -
                              yy.view(batch_size, -1))**2).mean()*batch_size
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        pred_train_mse /= (ntrain*(seq_inter_len-1))

        model.eval()

        pred_train_inter_mse = 0.0
        pred_train_extra_mse = 0.0
        pred_test_inter_mse = 0.0
        pred_test_extra_mse = 0.0

        if step_show:
            with torch.no_grad():
                for u_batch, modulations, coord_batch, idx in valid_loader:
                    u_batch = u_batch.cuda()
                    coord_batch = coord_batch.cuda()
                    input_frame = u_batch[..., 0]
                    batch_size = u_batch.shape[0]

                    for t in range(seq_extra_len+seq_inter_len - 1):
                        target_frame = u_batch[..., t+1]
                        pred = model(input_frame)
                        loss = myloss(pred.view(batch_size, -1),
                                        target_frame.view(batch_size, -1))
                        # test_l2 += loss.item()
                        xx = pred * sigma.cuda() + mean.cuda()
                        yy = target_frame * sigma.cuda() + mean.cuda()

                        # We don't use the first timesteps to compute the extrapolation loss
                        if t >= seq_inter_len:
                            pred_train_extra_mse += ((xx.view(batch_size, -1) -
                                                yy.view(batch_size, -1))**2).mean()*batch_size
                        if t < seq_inter_len:
                            pred_train_inter_mse += ((xx.view(batch_size, -1) -
                                                    yy.view(batch_size, -1))**2).mean()*batch_size

                        input_frame = pred.detach()

                for u_batch, modulations, coord_batch, idx in test_loader:
                    u_batch = u_batch.cuda()
                    coord_batch = coord_batch.cuda()
                    input_frame = u_batch[..., 0]
                    batch_size = u_batch.shape[0]

                    for t in range(seq_extra_len+seq_inter_len - 1):
                        target_frame = u_batch[..., t+1]
                        pred = model(input_frame)
                        loss = myloss(pred.view(batch_size, -1),
                                        target_frame.view(batch_size, -1))
                        # test_l2 += loss.item()
                        xx = pred * sigma.cuda() + mean.cuda()
                        yy = target_frame * sigma.cuda() + mean.cuda()

                        # We don't use the first timesteps to compute the extrapolation loss
                        if t >= seq_inter_len:
                            pred_test_extra_mse += ((xx.view(batch_size, -1) -
                                            yy.view(batch_size, -1))**2).mean()*batch_size
                        if t < seq_inter_len:
                            pred_test_inter_mse += ((xx.view(batch_size, -1) -
                                            yy.view(batch_size, -1))**2).mean()*batch_size
                        

                        input_frame = pred.detach()

                pred_train_inter_mse /= (ntrain*(seq_inter_len-1))
                pred_train_extra_mse /= (ntrain*(seq_extra_len-1))
                pred_test_inter_mse /= (ntest*(seq_inter_len-1))
                pred_test_extra_mse /= (ntest*(seq_extra_len-1))

        if step_show:
            wandb.log(
                {
                    "pred_train_inter_mse": pred_train_inter_mse,
                    "pred_train_extra_mse": pred_train_extra_mse,
                    'pred_test_inter_mse': pred_test_inter_mse,
                    'pred_test_extra_mse': pred_test_extra_mse,
                }
                )
        else:
            wandb.log(
                {
                    "pred_train_mse": pred_train_mse,
                },
                step=ep,
                commit=not step_show,
            )

        if pred_train_mse < best_loss:
            best_loss = pred_train_mse

            torch.save(
                {
                    "cfg": cfg,
                    "epoch": ep,
                    "fno": model.state_dict(),
                    "optimizer_inr": optimizer.state_dict(),
                    "loss": pred_train_mse,
                    "grid_tr": grid_tr,
                    "grid_te": grid_te,
                },
                f"{RESULTS_DIR}/{run_name}.pt",
            )

    t2 = default_timer()

    return None

if __name__ == "__main__":
    main()