"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from timeit import default_timer

import torch.nn.functional as F
from utilities3 import *

from coral.utils.data import (DatasetInputOutput, DatasetWithCode, get_data,
                              repeat_coordinates, set_seed, shape2coordinates,
                              subsample)

torch.manual_seed(0)
np.random.seed(0)
import scipy


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = (
            modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        )
        self.modes2 = modes2

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            self.scale
            * torch.rand(
                in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat
            )
        )

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2], self.weights2
        )

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels):
        super(MLP, self).__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)

    def forward(self, x):
        x = self.mlp1(x)
        x = F.gelu(x)
        x = self.mlp2(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9  # pad the domain if input is non-periodic

        self.p = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)
        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.mlp0 = MLP(self.width, self.width, self.width)
        self.mlp1 = MLP(self.width, self.width, self.width)
        self.mlp2 = MLP(self.width, self.width, self.width)
        self.mlp3 = MLP(self.width, self.width, self.width)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.q = MLP(self.width, 1, self.width * 4)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0, self.padding, 0, self.padding])

        x1 = self.conv0(x)
        x1 = self.mlp0(x1)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x1 = self.mlp1(x1)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x1 = self.mlp2(x1)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x1 = self.mlp3(x1)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., : -self.padding, : -self.padding]
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################

ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs * (ntrain // batch_size)

modes = 12
width = 32

# r = 5
# h = int(((421 - 1)/r) + 1)
# s = h

################################################################
# load data and data normalization
################################################################


# Data is of the shape (number of samples, grid size)
RESULTS_DIR = "/data/serrano/functa2functa/fno/navier-stokes/"
data_dir = "/data/serrano/deeponet-fourier-data"
dataset_name = "darcy"
same_grid = True
sub_tr = 0.05
sub_te = 4

(
    x_train,
    y_train,
    x_test,
    y_test,
    grid_inp_tr,
    grid_out_tr,
    grid_inp_te,
    grid_out_te,
) = get_data(data_dir, dataset_name, ntrain, ntest, sub_tr, sub_te, same_grid=same_grid)
print(
    f"data: {dataset_name}, x_tr: {x_train.shape}, y_tr: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}"
)
print(
    f"grid: grid_inp_tr: {grid_inp_tr.shape}, grid_out_tr: {grid_out_tr.shape}, grid_inp_te: {grid_inp_te.shape}, grid_out_te: {grid_out_te.shape}"
)

# interpolate input and output training

p = 105  # 221
grid_interpol_train = shape2coordinates([p, p]).flatten(0, 1)
x_interpol = torch.zeros((ntrain, p, p, 1))
y_interpol = torch.zeros((ntrain, p, p, 1))

print("begin interpolation")
print("grid pol", grid_interpol_train.shape)
print("grid inp", grid_inp_tr.shape)
print("", grid_inp_tr.shape)

for j in range(ntrain):
    print(j)
    x_inter = scipy.interpolate.griddata(
        grid_inp_tr, x_train[j], grid_interpol_train, method="linear"
    )
    x_nearest = scipy.interpolate.griddata(
        grid_inp_tr, x_train[j], grid_interpol_train, method="nearest"
    )
    x_inter[np.isnan(x_inter)] = x_nearest[np.isnan(x_inter)]
    x_interpol[j] = torch.tensor(x_inter.reshape(p, p, 1))

    y_inter = scipy.interpolate.griddata(
        grid_inp_tr, y_train[j], grid_interpol_train, method="linear"
    )
    y_nearest = scipy.interpolate.griddata(
        grid_inp_tr, y_train[j], grid_interpol_train, method="nearest"
    )
    y_inter[np.isnan(y_inter)] = y_nearest[np.isnan(y_inter)]
    y_interpol[j] = torch.tensor(y_inter.reshape(p, p, 1))

print("end interpolation")

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_interpol, y_interpol),
    batch_size=batch_size,
    shuffle=True,
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

myloss = LpLoss(size_average=False)
# y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)  # .reshape(batch_size, s, s)
        # out = y_normalizer.decode(out)
        # y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        loss.backward()

        optimizer.step()
        scheduler.step()
        train_l2 += loss.item()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x)  # .reshape(batch_size, s, s)
            # out = y_normalizer.decode(out)

            test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

torch.save(
    {"fno": model.state_dict(), "train_grid": grid_inp_tr},
    f"{RESULTS_DIR}/no_norm_reg_{sub_tr}.pt",
)
