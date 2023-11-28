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

from coral.utils.data.dynamics_dataset import TemporalDatasetWithCode
from coral.utils.data.load_data import get_dynamics_data, set_seed

torch.manual_seed(0)
np.random.seed(0)

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
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
        
        input: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 2 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width)
        # input channel is 12: the solution of the previous 10 timesteps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)

        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)
        self.bn1 = torch.nn.BatchNorm2d(self.width)
        self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

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
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


################################################################
# configs
################################################################
TRAIN_PATH = "/data/serrano/"

ntrain=512
ntest=32
batch_size = 20
learning_rate = 0.001
epochs = 500
iterations = epochs * (512 // batch_size)

modes = 12
#width = 20
width = 32

batch_size = 20
batch_size2 = batch_size

epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

# r = 5
# h = int(((421 - 1)/r) + 1)
# s = h

################################################################
# load data and data normalization
################################################################


# Data is of the shape (number of samples, grid size)
RESULTS_DIR = "/data/serrano/functa2functa/fno/navier-stokes/"
data_dir = "/data/serrano/"
dataset_name = "navier-stokes-dino"
same_grid = True
sub_tr = 1
sub_te = 4
sub_from=4
sequence_length=20

(u_train, u_test, grid_tr, grid_te) = get_dynamics_data(
        data_dir,
        dataset_name,
        ntrain,
        ntest,
        sequence_length=sequence_length,
        sub_tr=sub_tr,
        sub_te=sub_te,
        sub_from=sub_from,
        same_grid=same_grid,
    )

print(
    f"data: {dataset_name}, u_tr: {u_train.shape}, u_test: {u_test.shape}"
)
print(
    f"grid: grid_tr: {grid_tr.shape}, grid_inp_te: {grid_te.shape}"
)

trainset = TemporalDatasetWithCode(
        u_train, grid_tr, 32, dataset_name, None
    )
testset = TemporalDatasetWithCode(
        u_test, grid_te, 32, dataset_name, None
    )

ntrain = len(trainset)
ntest = len(testset)

mean, sigma = u_train.mean(), u_train.std()
u_train = (u_train - mean) / sigma
u_test = (u_test - mean) / sigma

#y_normalizer = UnitGaussianNormalizer(y_train)
#y_train = y_normalizer.encode(y_train)

# x_train = x_train.reshape(ntrain,s,s,1)
# x_test = x_test.reshape(ntest,s,s,1)

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
   testset, batch_size=batch_size, shuffle=False
)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for u_batch, modulations, coord_batch, idx in train_loader:
        u_batch = u_batch.cuda()
        coord_batch = coord_batch.cuda()
        input_frame = u_batch[..., 0]
        batch_size = u_batch.shape[0]
        
        loss = 0
        for t in range(sequence_length - 1):
            target_frame = u_batch[..., t+1]
            pred = model(input_frame)
            #print('device', pred.device, target_frame.device)
            #print('pred', pred.shape, target_frame.shape)
            loss += myloss(pred.view(batch_size, -1), target_frame.view(batch_size, -1))
            optimizer.zero_grad()
            #loss = ((pred - target_frame)**2).mean()
            input_frame = pred.detach()
            xx = pred * sigma.cuda() + mean.cuda()
            yy = target_frame * sigma.cuda() + mean.cuda()
            train_l2 += ((xx.view(batch_size, -1) - yy.view(batch_size, -1))**2).mean()*batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for u_batch, modulations, coord_batch, idx  in test_loader:
            u_batch = u_batch.cuda()
            coord_batch = coord_batch.cuda()
            input_frame = u_batch[..., 0]
            batch_size = u_batch.shape[0]

            for t in range(sequence_length - 1):
                target_frame = u_batch[..., t+1]
                pred = model(input_frame)
                loss = myloss(pred.view(batch_size, -1), target_frame.view(batch_size, -1))
                #test_l2 += loss.item()
                xx = pred * sigma.cuda() + mean.cuda()
                yy = target_frame * sigma.cuda() + mean.cuda()
                test_l2 += ((xx.view(batch_size, -1) - yy.view(batch_size, -1))**2).mean()*batch_size

                input_frame = pred.detach()

    train_l2 /= (ntrain*(sequence_length-1))
    test_l2 /= (ntest*(sequence_length-1))

    t2 = default_timer()
    print(ep, t2 - t1, train_l2, test_l2)

torch.save(
    {"fno": model.state_dict(), "train_grid": grid_tr},
    f"{RESULTS_DIR}/time_{sequence_length}_reg_{sub_tr}.pt",
)
