"""
@author: Zongyi Li and Daniel Zhengyu Huang
"""
from pathlib import Path
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from torch.optim import Adam, AdamW
from coral.utils.models.load_inr import create_inr_instance
from coral.utils.data.load_data import get_operator_data
from coral.utils.data.operator_dataset import OperatorDataset
from coral.losses import batch_mse_rel_fn
from coral.mfn import FourierNet, HyperMAGNET, HyperMultiscaleBACON
from coral.mlp import MLP, Derivative, ResNet
from coral.siren import ModulatedSiren
from coral.metalearning import outer_step
from torch.utils.data import DataLoader
from coral.metalearning import inner_loop_step
from coral.utils.data.load_modulations import load_operator_modulations

torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

################################################################
# configs
################################################################

data_dir = "/data/serrano/deeponet-fourier-data"
dataset_name = "airfoil"
ntrain = 1000
ntest = 200

# test
torch.set_default_dtype(torch.float32)

inr_run_name = "glowing-music-4181"
mlp_run_name = "olive-bee-4857"

root_dir = Path(os.getenv("WANDB_DIR")) / dataset_name

input_inr = torch.load(root_dir / "inr" / f"{inr_run_name}.pt")
input_model = torch.load(root_dir / "model" / f"{mlp_run_name}.pt")

# load inr weights
load_cfg = input_inr["cfg"]
latent_dim_in = input_inr["cfg"].inr_in.latent_dim
latent_dim_out = input_inr["cfg"].inr_out.latent_dim


x_train, y_train, x_test, y_test, grid_tr, grid_te = get_operator_data(
data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True)


grid = grid_tr[0].unsqueeze(0).to(torch.float32)

print('x_train', x_train.shape, x_train.dtype)
print('y_train', y_train.shape, y_train.dtype)
print('x_test', x_test.shape, x_test.dtype)
print('y_test', y_test.shape, y_test.dtype)
print('grid_tr', grid_tr.shape, grid_tr.dtype)
print('grid_te', grid_te.shape, grid_te.dtype)

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

################################################################
# design
################################################################
from scipy import optimize

# symmetrical 4-digit NASA airfoil
# the airfoil is in [0,1]
def NACA_shape(x, digit=12):
    return 5 * (digit / 100.0) * (
                0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)

# generate mesh between a and b
# dx0, dx0*r, dx0*r^2 ... dx0*r^{N-2}
# b - a = dx0*(r^{N-1} - 1)/(r - 1)
def GeoSpace(a, b, N, r=-1.0, dx0=-1.0):
    xx = np.linspace(a, b, N)
    if r > 1 or dx0 > 0:
        if r > 1:
            dx0 = (b - a) / ((r ** (N - 1) - 1) / (r - 1))
            dx = dx0
            for i in range(1, N - 1):
                xx[i] = xx[i - 1] + dx
                dx *= r
        else:
            # first use r=1.05 to generate half of the grids
            # then compute r and generate another half of the grids
            f = lambda r: (r - 1) * (b - a) - dx0 * (r ** (N - 1) - 1)
            r = optimize.bisect(f, 1 + 1e-4, 1.5)

            if r > 1.02:
                r = min(r, 1.02)
                dx = dx0
                Nf = 3 * N // 4

                for i in range(1, Nf):
                    xx[i] = xx[i - 1] + dx
                    dx *= r

                a = xx[Nf - 1]
                dx0 = dx

                f = lambda r: (r - 1) * (b - a) - dx0 * (r ** (N - Nf) - 1)
                r = optimize.bisect(f, 1 + 1e-4, 2.0)

                for i in range(Nf, N - 1):
                    xx[i] = xx[i - 1] + dx
                    dx *= r
            else:
                dx = dx0
                for i in range(1, N - 1):
                    xx[i] = xx[i - 1] + dx
                    dx *= r
    return xx

# Nx point on top skin
def NACA_shape_mesh(Nx, method="stretching", ratio=1.0):
    if method == "stretching":
        xx = np.zeros(Nx)
        xx[1:] = GeoSpace(0, 1, Nx - 1, r=ratio ** (1 / (Nx - 3)))
        xx[1] = xx[2] / 4.0
    else:
        print("method : ", method, " is not recognized")

    xx = xx[::-1]
    yy = np.hstack((NACA_shape(xx), -NACA_shape(xx[-2::-1])))
    xx = np.hstack((xx, xx[-2::-1]))
    return xx, yy


# The undeformed box is
# 0.5 - Lx/2  (8)        0.5 - Lx/6  (7)         0.5 + Lx/6  (6)          0.5 + Lx/2  (5)         (y = Ly/2)
#
# 0.5 - Lx/2  (1)        0.5 - Lx/6  (2)         0.5 + Lx/6  (3)          0.5 + Lx/2  (4)         (y = -Ly/2)
#
# basis function at node (i)   is Bᵢ   = Φᵢ(x) Ψ₁(y)    (1 ≤ i ≤ 4)
# basis function at node (i+4) is Bᵢ₊₄ = Φᵢ(x) Ψ₂(y)    (1 ≤ i ≤ 4)
#
# The map is
# (x, y) -> (x, y) + dᵢ Bᵢ(x,  y)
#
def NACA_sdesign(theta, x, y, Lx=1.5, Ly=0.2):
    x1, x2, x3, x4 = 0.5 - Lx / 2, 0.5 - Lx / 6, 0.5 + Lx / 6, 0.5 + Lx / 2
    y1, y2 = - Ly / 2, Ly / 2

    phi1 = (x - x2) * (x - x3) * (x - x4) / ((x1 - x2) * (x1 - x3) * (x1 - x4))
    phi2 = (x - x1) * (x - x3) * (x - x4) / ((x2 - x1) * (x2 - x3) * (x2 - x4))
    phi3 = (x - x1) * (x - x2) * (x - x4) / ((x3 - x1) * (x3 - x2) * (x3 - x4))
    phi4 = (x - x1) * (x - x2) * (x - x3) / ((x4 - x1) * (x4 - x2) * (x4 - x3))

    psi1 = (y - y2) / (y1 - y2)
    psi2 = (y - y1) / (y2 - y1)

    B = torch.stack([phi2 * psi1, phi3 * psi1, phi4 * psi1, phi4 * psi2, phi3 * psi2, phi2 * psi2, phi1 * psi2], dim=0)
    return x, y + torch.matmul(theta, B)

def Cgrid2Cylinder(cnx1, cnx2, cny, Cgrid, Cylinder):
    # Cgrid
    nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1

    for j in range(cny + 1):
        if j == 0:
            Cylinder[0:cnx1 + cnx2, j] = Cgrid[0:cnx1 + cnx2]
            Cylinder[cnx1 + cnx2:2 * nx1 + cnx2 - 1, j] = Cylinder[cnx1::-1, j]
        else:
            Cylinder[:, j] = Cgrid[(j - 1) * (2 * cnx1 + cnx2 + 1) + cnx1 + cnx2: (j - 1) * (
                        2 * cnx1 + cnx2 + 1) + cnx1 + cnx2 + 1 + 2 * cnx1 + cnx2]

def Cylinder2Cgrid(cnx1, cnx2, cny, Cylinder, Cgrid):
    # Cylinder,
    nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1

    for j in range(cny + 1):
        if j == 0:
            Cgrid[0:cnx1 + cnx2] = Cylinder[0:cnx1 + cnx2, j]
        else:
            Cgrid[(j - 1) * (2 * cnx1 + cnx2 + 1) + cnx1 + cnx2: (j - 1) * (
                        2 * cnx1 + cnx2 + 1) + cnx1 + cnx2 + 1 + 2 * cnx1 + cnx2] = Cylinder[:, j]

# c: number of cells
# cnx1 C mesh behind trailing edge
# cnx2 C mesh around airfoil
# cny radial direction
#
# The airfoil is in [0,1]
# R: radius of C mesh
# L: the right end of the mesh
# the bounding box of the mesh is [Rc-R, L], [-R, R]
#
# dy0, vertical mesh size
cnx1=50
dy0=2.0 / 120.0
cnx2=120
cny=50
R=40
Rc=1.0
L=40
cnx = 2 * cnx1 + cnx2
nx1, nx2, ny = cnx1 + 1, cnx2 + 1, cny + 1  # points
nnodes = (2 * nx1 + cnx2 - 1) * cny + (nx1 + cnx2 - 1)

xx_airfoil, yy_airfoil = NACA_shape_mesh(cnx2 // 2 + 1, method="stretching")
xx_inner = GeoSpace(0, 1, nx1, dx0=np.sqrt((xx_airfoil[0] - xx_airfoil[1]) ** 2 + (yy_airfoil[0] - yy_airfoil[1]) ** 2) / (L - 1))
xx_outer = GeoSpace(Rc, L, nx1)
wy = GeoSpace(0, 1, ny, dx0=dy0 / R)

xx_airfoil = torch.tensor(xx_airfoil, device='cuda', dtype=torch.float)
yy_airfoil = torch.tensor(yy_airfoil, device='cuda', dtype=torch.float)
xx_inner = torch.tensor(xx_inner, device='cuda', dtype=torch.float)
xx_outer = torch.tensor(xx_outer, device='cuda', dtype=torch.float)
wy = torch.tensor(wy, device='cuda', dtype=torch.float)

def Theta2Mesh(theta, xx_airfoil=xx_airfoil, yy_airfoil=yy_airfoil, xx_inner=xx_inner, xx_outer=xx_outer):
    # assert (len(theta) == 8 and theta[0] == 0.0)
    assert (len(theta) == 7)

    xx_airfoil, yy_airfoil = NACA_sdesign(theta, xx_airfoil, yy_airfoil)

    xy_inner = torch.zeros((2 * nx1 + cnx2 - 1, 2), device='cuda', dtype=torch.float)
    xy_outer = torch.zeros((2 * nx1 + cnx2 - 1, 2), device='cuda', dtype=torch.float)
    # top flat
    xy_inner[:nx1, 0] = torch.flip(xx_airfoil[0] * (1 - xx_inner) + L * xx_inner, dims=[0])
    xy_inner[:nx1, 1] = torch.flip(yy_airfoil[0] * (1 - xx_inner), dims=[0])
    xy_outer[:nx1, 0] = torch.flip(xx_outer, dims=[0])
    xy_outer[:nx1, 1] = R

    # airfoil
    xy_inner[nx1 - 1:nx1 + cnx2, 0] = xx_airfoil
    xy_inner[nx1 - 1:nx1 + cnx2, 1] = yy_airfoil

    θθ = torch.linspace(np.pi / 2, 3 * np.pi / 2, nx2)
    xy_outer[nx1 - 1:nx1 + cnx2, 0] = R * torch.cos(θθ) + Rc
    xy_outer[nx1 - 1:nx1 + cnx2, 1] = R * torch.sin(θθ)
    # bottom flat
    xy_inner[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 0] = torch.flip(xy_inner[:nx1, 0], dims=[0])
    xy_inner[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 1] = torch.flip(xy_inner[:nx1, 1], dims=[0])
    xy_outer[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 0] = xx_outer
    xy_outer[nx1 + cnx2 - 1:2 * nx1 + cnx2 - 1, 1] = -R

    # Construct Cylinder grid
    xx_Cylinder = torch.outer(xy_inner[:, 0], 1 - wy) + torch.outer(xy_outer[:, 0], wy)
    yy_Cylinder = torch.outer(xy_inner[:, 1], 1 - wy) + torch.outer(xy_outer[:, 1], wy)
    out = torch.stack([xx_Cylinder, yy_Cylinder], dim=-1).unsqueeze(0)
    return out, xx_Cylinder, yy_Cylinder

def compute_F(XC, YC, p, cnx1=50, cnx2=120, cny=50):
    p = p.squeeze()
    xx, yy, p = XC[cnx1:-cnx1, 0], YC[cnx1:-cnx1, 0], p[cnx1:-cnx1, 0]

    drag = torch.matmul(yy[0:cnx2]-yy[1:cnx2+1], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    lift = torch.matmul(xx[1:cnx2+1]-xx[0:cnx2], (p[0:cnx2] + p[1:cnx2+1])/2.0)
    return drag, lift

################################################################
# inverse optimization
################################################################

#model = torch.load(PATH + '/model/naca_p_w32_500')

modes = 12
width = 32

r1 = 1
r2 = 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

input_dim = 2
output_dim_in = 2
output_dim_out = 1

load_cfg.inr = load_cfg.inr_in
inr_in = create_inr_instance(
    load_cfg, input_dim=input_dim, output_dim=output_dim_in, device="cuda"
)
inr_in = inr_in.cuda()
inr_in.load_state_dict(input_inr["inr_in"])
inr_in.eval()
alpha_in = input_inr['alpha_in']

#modulations = torch.zeros((1, latent_dim_in), dtype=torch.float32).cuda().requires_grad_(True)
#print('modulation', modulations.dtype)
#print('grid', grid.dtype)
#print('alpha_in', alpha_in.dtype)
#for step in range(3):
#    modulations = inner_loop_step(inr_in,
#            modulations,
#            grid.cuda(),
#            xx.cuda(),
#            alpha_in.cuda(),
#            is_train=True,
#            gradient_checkpointing=False,
#            loss_type="mse",)

#a_pred = inr_in.modulated_forward(grid.cuda(), modulations)
#tmp_loss = ((a_pred - xx)**2).mean()
#print('tototottoto tmp_loss', tmp_loss)



print('inr_in weight', inr_in.modulation_net.net.weight.dtype)


load_cfg.inr = load_cfg.inr_out
inr_out = create_inr_instance(
    load_cfg, input_dim=input_dim, output_dim=output_dim_out, device="cuda"
)
inr_out.load_state_dict(input_inr["inr_out"])
inr_out.eval()
inr_out = inr_out.cuda()
alpha_out = input_inr['alpha_out']

modulations_dir = Path(os.getenv("WANDB_DIR")) / dataset_name / "modulations"
load_run_name = inr_run_name
inner_steps = 3

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
    latent_dim_a=load_cfg.inr_in.latent_dim,
    latent_dim_u=load_cfg.inr_out.latent_dim,
    dataset_name=None,
    data_to_encode=None,
)

testset = OperatorDataset(x_test,
    y_test,
    grid_te,
    latent_dim_a=load_cfg.inr_in.latent_dim,
    latent_dim_u=load_cfg.inr_out.latent_dim,
    dataset_name=None,
    data_to_encode=None,
)
ntrain = len(trainset)
ntest = len(testset)

train_loader = DataLoader(dataset=trainset, batch_size=1, shuffle=False)
test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False)


xmax = 40
xmin = -39
ymax = 40
ymin = -40

for substep, (a_s, u_s, za_s, zu_s, coords, idx) in enumerate(test_loader):

    XC = a_s[..., 0]*(xmax - xmin) + xmin
    YC = a_s[..., 0]*(ymax - ymin) + ymin

    drag, lift = compute_F(XC.squeeze(0), YC.squeeze(0), u_s.squeeze(0))

    print(idx, 'drag', drag, 'lift', lift, (drag/lift)**2)

    if lift<-0.30:
        x = a_s
        ind = -1
        X = x[ind, :, :, 0].squeeze().detach().cpu().numpy()
        Y = x[ind, :, :, 1].squeeze().detach().cpu().numpy()
        pred = u_s[ind].squeeze().detach().cpu().numpy()
        nx = 40//r1
        ny = 20//r2
        X_small = X[nx:-nx, :ny]
        Y_small = Y[nx:-nx, :ny]

        pred_small = pred[nx:-nx, :ny]
        fig, ax = plt.subplots(ncols=2,  figsize=(16, 8))
        c0 = ax[0].pcolormesh(X, Y, pred, shading='gouraud', rasterized=True)
        fig.colorbar(c0, ax=ax[0], label='Volume')
        c1 = ax[1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud', rasterized=True)
        fig.colorbar(c1, ax=ax[1], label='Near the surface')

        plt.savefig(f"example_drag_{drag}_lift_{lift}_idx_{idx}.pdf", dpi=50)
