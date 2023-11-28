import glob
import os
import random
import shelve
from pathlib import Path

import einops
import h5py
import numpy as np
import scipy.io
import torch
import torch.nn.functional as F
import xarray as xr
from scipy import io
from torch.utils.data import Dataset


def get_data(data_dir, dataset_name, ntrain, ntest, sub_tr=1, sub_te=1, same_grid=True):
    """Get training and test data as well as associated coordinates, depending on the dataset name.

    Args:
        data_dir (str): path to the dataset directory
        dataset_name (str): dataset name (e.g. "navier-stokes)
        ntrain (int): number of training samples
        ntest (int): number of test samples
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_tr]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_tr*len(x)). Defaults to 1.
        sub_tr (int or float, optional): when set to int > 1, subsamples x as x[::sub_te]. When set to float < 1, subsamples x as x[index] where index is a random index of shape int(sub_te*len(x)). Defaults to 1.
        same_grid (bool, optional): If True, input and output grid should be the same. Defaults to True.

    Raises:
        NotImplementedError: _description_

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
        grid_inp_tr (torch.Tensor): coordinates of x_train
        grid_out_tr (torch.Tensor): coordinates of y_train
        grid_inp_te (torch.Tensor): coordinates of x_test
        grid_out_te (torch.Tensor): coordinates of y_test
    """

    data_dir = Path(data_dir)

    if dataset_name == "burgers":
        min_sub = 8  # 4 -> 2048  # 1024
        x_train, y_train, x_test, y_test = get_burgers(
            data_dir / "burgers_data_R10.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "darcy":
        min_sub = 1
        x_train, y_train = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth1.mat", ntrain
        )
        x_test, y_test = get_darcy_pwc(
            data_dir / "piececonst_r421_N1024_smooth2.mat", ntest
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

        # x_max, x_min = x_train.max(), x_train.min()
        # x_train, x_test = (
        #    (x_train - x_min) / (x_max - x_min),
        #    (x_test - x_min) / (x_max - x_min),
        # )
        mu, sigma = x_train.mean(), x_train.std()
        x_train, x_test = (x_train - mu) / sigma, (x_test - mu) / sigma

    elif dataset_name == "kdv":
        min_sub = 8  # 8 -> 1024
        x_train, y_train, x_test, y_test = get_kdv(
            data_dir / "kdv_train_test.mat", ntrain, ntest
        )
        x_train, y_train = x_train[:, ::min_sub], y_train[:, ::min_sub]
        x_test, y_test = x_test[:, ::min_sub], y_test[:, ::min_sub]

    elif dataset_name == "navier-stokes":
        min_sub = 1
        reader = MatReader(data_dir / "ns_V1e-3_N5000_T50.mat")
        u = reader.read_field("u")

        u_train = u[:ntrain, :]
        u_test = u[-ntest:, :]

        x_train, y_train = u_train[..., 9], u_train[..., 19]  # 19 previously
        x_test, y_test = u_test[..., 9], u_test[..., 19]  # 19 previously

    elif dataset_name == "navier-stokes-256":
        print("naviertokes256")
        min_sub = 2  # 2 usually
        # train_dir = str(data_dir) + "/" + "navier_1e-3_256_2_train.shelve"
        # test_dir = str(data_dir) + "/" + "navier_1e-3_256_2_test.shelve"

        # data_train = dict(shelve.open(train_dir))
        # data_test = dict(shelve.open(test_dir))

        shelve_dir = str(data_dir) + "/" + "ns_1e-3_1200.shelve"
        data = dict(shelve.open(shelve_dir))

        data.pop("a")
        data.pop("t")

        # concatenate dictionary to be of shape (ntrain, 40, 256, 256)

        u = torch.tensor(
            np.concatenate(
                list(
                    map(lambda key: np.expand_dims(
                        np.array(data[key]), 0), data.keys())
                )
            )
        )

        u_train = u[:1000]
        u_test = u[-200:]

        # select the 9 and 19 indices

        x_train, y_train = u_train[..., 9], u_train[..., 19]
        x_test, y_test = u_test[..., 9], u_test[..., 19]

        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[:, ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    elif dataset_name == "advection":
        min_sub = 1
        x_train, y_train = get_advection(data_dir / "train_IC2.npz", ntrain)
        x_test, y_test = get_advection(data_dir / "test_IC2.npz", ntest)

    elif dataset_name == "pipe":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_pipe(
            data_dir, ntrain, ntest, min_sub=min_sub
        )

    elif dataset_name == "airfoil":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_airfoil(
            data_dir, ntrain, ntest, min_sub=min_sub
        )

    elif dataset_name == "elasticity":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_elasticity(
            data_dir, ntrain, ntest, min_sub=min_sub
        )

    elif dataset_name == "shallow-water":
        min_sub = 1
        x_train, y_train, x_test, y_test = get_shallow_water(
            data_dir, ntrain, ntest, min_sub=1
        )
        x_train, y_train = (
            x_train[:, ::min_sub, ::min_sub],
            y_train[..., ::min_sub, ::min_sub],
        )
        x_test, y_test = (
            x_test[:, ::min_sub, ::min_sub],
            y_test[:, ::min_sub, ::min_sub],
        )

    else:
        raise NotImplementedError

    # expects x_train, y_train, x_test, y_test to be of shape (N, dx, 1) or (N, dx1, dx2, 1)

    if dataset_name in ["elasticity"]:
        average_grid = x_train.mean(0)
        grid_inp_tr = average_grid
        grid_inp_te = average_grid
        grid_out_tr = average_grid
        grid_out_te = average_grid

    elif dataset_name in ["shallow-water"]:
        grid_inp_tr = shape2spherical_coordinates(x_train.shape[1:-1])
        grid_inp_te = shape2spherical_coordinates(x_train.shape[1:-1])
        grid_out_tr = shape2spherical_coordinates(x_train.shape[1:-1])
        grid_out_te = shape2spherical_coordinates(x_train.shape[1:-1])

    else:
        x_train = x_train.unsqueeze(-1)
        x_test = x_test.unsqueeze(-1)

        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

        # create grid for the high resolution data
        grid_inp_tr = shape2coordinates(x_train.shape[1:-1])
        grid_out_tr = shape2coordinates(y_train.shape[1:-1])
        grid_inp_te = shape2coordinates(x_test.shape[1:-1])
        grid_out_te = shape2coordinates(y_train.shape[1:-1])

    # subample data and grid, either uniformly (sub is int) or not (sub is float)

    if isinstance(sub_tr, int):
        grid_inp_tr = subsample(grid_inp_tr, sub_tr)
        grid_out_tr = subsample(grid_out_tr, sub_tr)
        x_train = subsample(x_train, sub_tr, missing_batch=False)
        y_train = subsample(y_train, sub_tr, missing_batch=False)

    if isinstance(sub_te, int):
        grid_inp_te = subsample(grid_inp_te, sub_te)
        grid_out_te = subsample(grid_out_te, sub_te)
        x_test = subsample(x_test, sub_te, missing_batch=False)
        y_test = subsample(y_test, sub_te, missing_batch=False)

    if isinstance(sub_tr, float) and (sub_tr < 1):
        N = x_train.shape[0]
        C = x_train.shape[-1]
        perm = torch.randperm(x_train.reshape(N, -1, C).shape[1])
        mask_tr = perm[: int(sub_tr * len(perm))].clone().sort()[0]

        if same_grid and x_train.ndim == y_train.ndim:
            mask_tr_out = mask_tr.clone().sort()[0]
            print("same_grid for out")
        else:
            perm = torch.randperm(y_train.reshape(N, -1, C).shape[1])
            mask_tr_out = perm[: int(sub_tr * len(perm))].clone().sort()[0]

        grid_inp_tr = subsample(grid_inp_tr, mask_tr)
        grid_out_tr = subsample(grid_out_tr, mask_tr_out)
        x_train = subsample(x_train, mask_tr, missing_batch=False)
        y_train = subsample(y_train, mask_tr_out, missing_batch=False)

    if isinstance(sub_te, float) and (sub_te < 1):
        N = x_test.shape[0]
        C = x_test.shape[-1]
        perm = torch.randperm(x_test.reshape(N, -1, C).shape[1])
        mask_te = perm[: int(sub_te * len(perm))].clone().sort()[0]

        if same_grid and x_test.ndim == y_test.ndim:
            mask_te_out = mask_te.clone().sort()[0]

        else:
            perm = torch.randperm(y_test.reshape(N, -1, C).shape[1])
            mask_te_out = perm[: int(sub_te * len(perm))].clone().sort()[0]

        grid_inp_te = subsample(grid_inp_te, mask_te)
        grid_out_te = subsample(grid_out_te, mask_te_out)
        x_test = subsample(x_test, mask_te, missing_batch=False)
        y_test = subsample(y_test, mask_te_out, missing_batch=False)

    return (
        x_train,
        y_train,
        x_test,
        y_test,
        grid_inp_tr,
        grid_out_tr,
        grid_inp_te,
        grid_out_te,
    )


def get_kdv(filename, ntrain, ntest):
    """Get kdv data.

    Args:
        filename (str or Path): path to dataset
        ntrain (int): number of training samples
        ntest (int): number of test samples

    Returns:
        x_train (torch.Tensor): (ntrain, ..., 1)
        y_train (torch.Tensor): (ntrain, ..., 1)
        x_test (torch.Tensor): (ntest, ..., 1)
        y_test (torch.Tensor): (ntest, ..., 1)
    """
    rw_ = io.loadmat(filename)
    x_data = rw_["input"].astype(np.float32)
    y_data = rw_["output"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :]
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :]

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_advection(filename, ndata):
    """Get advection data.

    Args:
        filename (str or Path): path to dataset

    Returns:
        x (torch.Tensor)
        y (torch.Tensor)
    """
    data = np.load(filename)
    x = data["x"].astype(np.float32)
    t = data["t"].astype(np.float32)
    u = data["u"].astype(np.float32)  # (N, nt, nx)
    N = u.shape[0]

    x = u[:, 0, :].copy()  # (N, nx)
    y = u[:, 1:, :].copy()  # (N, (nt-1), nx)

    x = x[:ndata, :]  # (ndata, nx)
    y = y[:ndata, :, :]  # (ndata, (nt-1), nx)

    x = torch.from_numpy(x)
    y = torch.from_numpy(y)

    return x, y


def get_darcy_pwc(filename, ndata):
    """Get darcy data.

    Args:
        filename (str or Path): path to dataset
        ndata (int): number of samples to return

    Returns:
        x (torch.Tensor): (ndata, ..., 1)
        y (torch.Tensor): (ndata, ..., 1)
    """
    # 5->85x85, 6->71x71, 7->61x61, 10->43x43, 12->36x36, 14->31x31, 15->29x29

    # Data is of the shape (number of samples = 1024, grid size = 421x421)
    data = io.loadmat(filename)
    x_branch = data["coeff"][:ndata, :, :].astype(np.float32) * 0.1 - 0.75
    y = data["sol"][:ndata, :, :].astype(np.float32) * 100
    # The dataset has a mistake that the BC is not 0.
    y[:, 0, :] = 0
    y[:, -1, :] = 0
    y[:, :, 0] = 0
    y[:, :, -1] = 0

    x_branch = torch.from_numpy(x_branch)
    y = torch.from_numpy(y)

    return x_branch, y


def get_burgers(filename, ntrain, ntest):
    """Get burgers data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)
    data = io.loadmat(filename)
    x_data = data["a"].astype(np.float32)
    y_data = data["u"].astype(np.float32)
    x_train = x_data[:ntrain, :]
    y_train = y_data[:ntrain, :].astype(np.float32)  # added
    x_test = x_data[-ntest:, :]
    y_test = y_data[-ntest:, :].astype(np.float32)  # added

    x_train = torch.from_numpy(x_train)
    x_test = torch.from_numpy(x_test)
    y_train = torch.from_numpy(y_train)
    y_test = torch.from_numpy(y_test)

    return x_train, y_train, x_test, y_test


def get_pipe(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    INPUT_X = os.path.join(filename, "../pipe/Pipe_X.npy")
    INPUT_Y = os.path.join(filename, "../pipe/Pipe_Y.npy")
    OUTPUT_Sigma = os.path.join(filename, "../pipe/Pipe_Q.npy")

    N = ntrain + ntest
    r1 = min_sub
    r2 = min_sub
    s1 = int(((129 - 1) / r1) + 1)
    s2 = int(((129 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    # input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 0]
    output = torch.tensor(output, dtype=torch.float)

    x_train = inputY[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:N][:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = inputY[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[:N][-ntest:, ::r1, ::r2][:, :s1, :s2]
    # x_train = x_train.reshape(ntrain, s1, s2, 2)
    # x_test = x_test.reshape(ntest, s1, s2, 2)

    xmax = torch.max(x_train)
    xmin = torch.min(x_train)
    x_train = (x_train - xmin) / (xmax - xmin)
    x_test = (x_test - xmin) / (xmax - xmin)

    return x_train, y_train, x_test, y_test


def get_airfoil(filename, ntrain, ntest, min_sub=1):
    """Get pipe data.

     Args:
         filename (str or Path): path to dataset
         ntrain (int): number of training samples
         ntest (int): number of test samples

    Returns:
         x_train (torch.Tensor): (ntrain, ..., 1)
         y_train (torch.Tensor): (ntrain, ..., 1)
         x_test (torch.Tensor): (ntest, ..., 1)
         y_test (torch.Tensor): (ntest, ..., 1)
    """

    # Data is of the shape (number of samples = 2048, grid size = 2^13)

    INPUT_X = os.path.join(filename, "../airfoil/naca/NACA_Cylinder_X.npy")
    INPUT_Y = os.path.join(filename, "../airfoil/naca/NACA_Cylinder_Y.npy")
    OUTPUT_Sigma = os.path.join(
        filename, "../airfoil/naca/NACA_Cylinder_Q.npy")

    r1 = min_sub
    r2 = min_sub
    s1 = int(((221 - 1) / r1) + 1)
    s2 = int(((51 - 1) / r2) + 1)

    inputX = np.load(INPUT_X)
    inputX = torch.tensor(inputX, dtype=torch.float)
    inputY = np.load(INPUT_Y)
    inputY = torch.tensor(inputY, dtype=torch.float)
    # input = torch.stack([inputX, inputY], dim=-1)

    output = np.load(OUTPUT_Sigma)[:, 4]
    output = torch.tensor(output, dtype=torch.float)

    x_train = inputY[:ntrain, ::r1, ::r2][:, :s1, :s2]
    y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
    x_test = inputY[-ntest:, ::r1, ::r2][:, :s1, :s2]
    y_test = output[-ntest:, ::r1, ::r2][:, :s1, :s2]
    # x_train = x_train.reshape(ntrain, s1, s2, 2)
    # x_test = x_test.reshape(ntest, s1, s2, 2)

    print(x_train.shape)

    xmax = torch.max(x_train)
    xmin = torch.min(x_train)
    x_train = (x_train - xmin) / (xmax - xmin)
    x_test = (x_test - xmin) / (xmax - xmin)

    return x_train, y_train, x_test, y_test


def get_elasticity(filename, ntrain, ntest, min_sub=1):
    PATH_Sigma = os.path.join(
        filename, "../elasticity/Random_UnitCell_sigma_10.npy")
    PATH_XY = os.path.join(filename, "../elasticity/Random_UnitCell_XY_10.npy")
    PATH_rr = os.path.join(filename, "../elasticity/Random_UnitCell_rr_10.npy")
    PATH_theta = os.path.join(
        filename, "../elasticity/Random_UnitCell_theta_10.npy")

    input_rr = np.load(PATH_rr)
    input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1, 0)
    input_s = np.load(PATH_Sigma)
    input_s = torch.tensor(
        input_s, dtype=torch.float).permute(1, 0).unsqueeze(-1)
    input_xy = np.load(PATH_XY)
    input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2, 0, 1)
    input_theta = np.load(PATH_theta)
    input_theta = torch.tensor(input_theta, dtype=torch.float).permute(1, 0)

    train_rr = input_rr[:ntrain]
    test_rr = input_rr[-ntest:]
    train_s = input_s[:ntrain]
    test_s = input_s[-ntest:]
    train_xy = input_xy[:ntrain]
    test_xy = input_xy[-ntest:]

    train_xy_sorted = []
    test_xy_sorted = []
    train_s_sorted = []
    test_s_sorted = []

    # for j in range(ntrain):
    #    ind = np.lexsort((train_xy[j][:,1],train_xy[j][:,0]))
    #    train_xy_sorted.append(train_xy[j][ind].unsqueeze(0))
    #    train_s_sorted.append(train_s[j][ind].unsqueeze(0))

    # for j in range(ntest):
    #    ind = np.lexsort((test_xy[j][:,1],test_xy[j][:,0]))
    #    test_xy_sorted.append(test_xy[j][ind].unsqueeze(0))
    #    test_s_sorted.append(test_s[j][ind].unsqueeze(0))

    # train_xy_sorted = torch.cat(train_xy_sorted, axis=0)
    # train_s_sorted = torch.cat(train_s_sorted, axis=0)
    # test_xy_sorted = torch.cat(test_xy_sorted, axis=0)
    # test_s_sorted = torch.cat(test_s_sorted, axis=0)

    sigma = train_s.std()
    mu = 0  # train_s.mean()

    train_s = (train_s - mu) / sigma
    test_s = (test_s - mu) / sigma

    return train_xy, train_s, test_xy, test_s


def get_shallow_water(filename, ntrain, ntest, min_sub=1):
    path_to_file = os.path.join(
        filename, "../shallow_water/data_t0180_freq6_N1200.h5")
    rf = h5py.File(path_to_file, "r")

    # shape (N, T, long, lat)
    # shape (1200, 6, 256, 128)

    initial_time = 0
    target_time = 5

    height_scale = 3 * 1e3
    vorticity_scale = 2

    height = torch.Tensor(rf["height"][()])
    vorticity = torch.Tensor(rf["vorticity"][()])

    # permute long and lat
    # create an extra dimension
    height = (height_scale * height).permute(0, 1, 3, 2).unsqueeze(-1)
    vorticity_scale = (vorticity_scale * vorticity).permute(0,
                                                            1, 3, 2).unsqueeze(-1)

    x_train = torch.cat(
        [height[:ntrain, initial_time], vorticity_scale[:ntrain, initial_time]], axis=-1
    )
    y_train = torch.cat(
        [height[:ntrain, target_time], vorticity_scale[:ntrain, target_time]], axis=-1
    )
    x_test = torch.cat(
        [height[-ntest:, initial_time], vorticity_scale[-ntest:, initial_time]], axis=-1
    )
    y_test = torch.cat(
        [height[-ntest:, target_time], vorticity_scale[-ntest:, target_time]], axis=-1
    )

    return x_train, y_train, x_test, y_test


class MatReader(object):
    """Loader for navier-stokes data"""

    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except BaseException:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x


def set_seed(seed=33):
    """Set all seeds for the experiments.

    Args:
        seed (int, optional): seed for pseudo-random generated numbers.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def subsample(x, sub=1, missing_batch=True):
    """Subsample data and coordinates in the same way.

    Args:
        x (torch.Tensor): data to be subsampled.
        sub (int or Tensor, optional): When set to int, subsamples x as x[::sub]. When set to Tensor of indices, slices x in the 1st dim. Defaults to 1.
        missing_batch (bool, optional): Coordinates are missing batch dimension at this stage and should be aligned with data wehn set to True. Defaults to True.

    Returns:
        x (torch.Tensor): subsampled array.
    """

    if missing_batch:
        x = x.unsqueeze(0)
    if isinstance(sub, int):
        # regular slicing
        if x.ndim == 3:
            x = x[:, ::sub]
        if x.ndim == 4:
            x = x[:, ::sub, ::sub]

    if isinstance(sub, torch.Tensor):
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = x[:, sub]

    if missing_batch:
        x = x.squeeze(0)
    return x


def shape2coordinates(spatial_shape):
    """Create coordinates from a spatial shape.

    Args:
        spatial_shape (list): Shape of data, i.e. [64, 64] for navier-stokes.

    Returns:
        grid (torch.Tensor): Coordinates that span (0, 1) in each dimension.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(0.0, 1.0, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def shape2multicoordinates(kernel_shape, spatial_shape, conv_kernel=3):
    coords = []
    for i in range(len(spatial_shape)):
        # quotient = torch.linspace(0.0, 1.0, spatial_shape[i]//(kernel_shape[i]-1)).repeat(kernel_shape[i]-1)
        # rest = torch.linspace(0.0, 1.0, spatial_shape[i]%(kernel_shape[i]-1))
        quotient = torch.linspace(
            0.0, 1.0, spatial_shape[i] // (kernel_shape[i] - conv_kernel)
        ).repeat(kernel_shape[i] - conv_kernel)
        rest = torch.linspace(
            0.0, 1.0, spatial_shape[i] % (kernel_shape[i] - conv_kernel)
        )
        coords.append(torch.cat([quotient, rest], axis=0))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.
    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return


class DatasetWithCode(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, latent_dim=256, with_time=False, sigma=0):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """
        if with_time:
            # input of the shape (B, ..., T)
            self.v = einops.rearrange(v, "b ... t -> (b t) ...")
            if sigma == 0:
                self.z = torch.zeros((self.v.shape[0], latent_dim))
            else:
                self.z = torch.randn((self.v.shape[0], latent_dim)) * sigma

            self.c = repeat_coordinates(grid, self.v.shape[0]).clone()

        else:
            self.v = v
            self.z = torch.zeros((v.shape[0], latent_dim))
            self.c = repeat_coordinates(grid, v.shape[0]).clone()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class ConvDatasetWithCode(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates."""

    def __init__(self, v, grid, kernel_dim, latent_dim=256, with_time=False, sigma=0):
        """
        Args:
            v (torch.Tensor): Dataset values, either x or y
            grid (torch.Tensor): Coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
            with_time (bool, optional): If True, time dimension should be flattened into batch dimension. Defaults to False.
        """
        if with_time:
            # input of the shape (B, ..., T)
            self.v = einops.rearrange(v, "b ... t -> (b t) ...")
            if sigma == 0:
                self.z = torch.zeros((self.v.shape[0], latent_dim))
            else:
                self.z = torch.randn((self.v.shape[0], latent_dim)) * sigma

            self.c = repeat_coordinates(grid, self.v.shape[0]).clone()

        else:
            self.v = v
            self.z = torch.zeros(
                (self.v.shape[0], latent_dim, *([kernel_dim] * grid.shape[-1]))
            )
            self.c = repeat_coordinates(grid, v.shape[0]).clone()

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class DatasetInputOutput(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(self, x, y, grid_x, grid_y, latent_dim=256, sigma=0, concat="batch"):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """

        if concat == "batch":
            self.v = torch.cat([x, y], axis=0)
            # self.z = torch.zeros((self.v.shape[0], latent_dim))
            if sigma == 0:
                self.z = torch.zeros((self.v.shape[0], latent_dim))
            else:
                self.z = torch.randn((self.v.shape[0], latent_dim)) * sigma
            self.c = torch.cat(
                [
                    repeat_coordinates(grid_x, x.shape[0]),
                    repeat_coordinates(grid_y, y.shape[0]),
                ],
                axis=0,
            )
        elif concat == "time":
            self.v = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], axis=-1)
            # self.z = torch.zeros((self.v.shape[0], latent_dim))
            if sigma == 0:
                self.z = torch.zeros((self.v.shape[0], latent_dim, 2))
            else:
                self.z = torch.randn((self.v.shape[0], latent_dim, 2)) * sigma
            self.c = torch.cat(
                [
                    repeat_coordinates(grid_x, x.shape[0]).unsqueeze(-1),
                    repeat_coordinates(grid_y, y.shape[0]).unsqueeze(-1),
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"invalid concat direction, {concat}")

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class ConvDatasetInputOutput(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(self, x, y, grid_x, grid_y, kernel_dim, latent_dim=8, concat="batch"):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """

        if concat == "batch":
            self.v = torch.cat([x, y], axis=0)
            # self.z = torch.zeros((self.v.shape[0], latent_dim))
            self.z = torch.zeros((self.v.shape[0], latent_dim))
            self.c = torch.cat(
                [
                    repeat_coordinates(grid_x, x.shape[0]),
                    repeat_coordinates(grid_y, y.shape[0]),
                ],
                axis=0,
            )
        elif concat == "time":
            self.v = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], axis=-1)
            # self.z = torch.zeros((self.v.shape[0], latent_dim))
            self.z = torch.zeros(
                (self.v.shape[0], latent_dim, *
                 ([kernel_dim] * grid_x.shape[-1]), 2)
            )
            # self.z = torch.randn((self.v.shape[0], latent_dim, *([kernel_dim]*grid_x.shape[-1]), 2)) * sigma

            self.c = torch.cat(
                [
                    repeat_coordinates(grid_x, x.shape[0]).unsqueeze(-1),
                    repeat_coordinates(grid_y, y.shape[0]).unsqueeze(-1),
                ],
                axis=-1,
            )
        else:
            raise ValueError(f"invalid concat direction, {concat}")

    def __len__(self):
        return len(self.v)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_v = self.v[idx, ...]
        sample_z = self.z[idx, ...]
        sample_c = self.c[idx, ...]

        return sample_v, sample_z, sample_c, idx

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class OperatorDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        a,
        u,
        grid_a,
        grid_u,
        latent_dim_in=256,
        latent_dim=256,
        sigma=0,
        concat="batch",
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        self.a = a
        self.u = u
        self.grid_a = repeat_coordinates(grid_a, a.shape[0])
        self.grid_u = repeat_coordinates(grid_u, u.shape[0])

        if sigma == 0:
            self.z_a = torch.zeros((self.a.shape[0], latent_dim_in))
            self.z_u = torch.zeros((self.u.shape[0], latent_dim))
        else:
            self.z_a = torch.randn((self.a.shape[0], latent_dim_in)) * sigma
            self.z_u = torch.randn((self.u.shape[0], latent_dim)) * sigma

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_a = self.a[idx, ...]
        sample_u = self.u[idx, ...]
        sample_z_a = self.z_a[idx, ...]
        sample_z_u = self.z_u[idx, ...]
        sample_grid_a = self.grid_a[idx, ...]
        sample_grid_u = self.grid_u[idx, ...]

        return (
            sample_a,
            sample_u,
            sample_z_a,
            sample_z_u,
            sample_grid_a,
            sample_grid_u,
            idx,
        )

    def __setitem__(self, new_z_a, new_z_u, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z_a[idx, ...] = new_z_a
        self.z_u[idx, ...] = new_z_u


class ConvOperatorDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        a,
        u,
        grid_a,
        grid_u,
        kernel_dim=8,
        latent_dim=8,
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        self.a = a
        self.u = u
        self.grid_a = repeat_coordinates(grid_a, a.shape[0])
        self.grid_u = repeat_coordinates(grid_u, u.shape[0])

        self.z_a = torch.zeros(
            (self.a.shape[0], latent_dim, *([kernel_dim] * grid_a.shape[-1]))
        )
        self.z_u = torch.zeros(
            (self.u.shape[0], latent_dim, *([kernel_dim] * grid_a.shape[-1]))
        )

    def __len__(self):
        return len(self.a)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample_a = self.a[idx, ...]
        sample_u = self.u[idx, ...]
        sample_z_a = self.z_a[idx, ...]
        sample_z_u = self.z_u[idx, ...]
        sample_grid_a = self.grid_a[idx, ...]
        sample_grid_u = self.grid_u[idx, ...]

        return (
            sample_a,
            sample_u,
            sample_z_a,
            sample_z_u,
            sample_grid_a,
            sample_grid_u,
            idx,
        )

    def __setitem__(self, new_z_a, new_z_u, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z_a[idx, ...] = new_z_a
        self.z_u[idx, ...] = new_z_u


def save_latent_dataset(filename, u, v, z):
    np.savez(filename, u=u, v=v, z=z)


def load_latent_dataset(filename):
    data = np.load(filename)
    return torch.tensor(data["u"]), torch.tensor(data["v"]), torch.tensor(data["z"])


class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


T_MIN = 202.66
T_MAX = 320.93


class ERA5Dataset(Dataset):
    """ERA5 temperature dataset.
    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """

    def __init__(
        self,
        path_to_data,
        inp_filepaths,
        out_filepaths,
        latent_dim,
        transform=None,
        normalize=True,
    ):
        # self.path_to_data = path_to_data
        self.transform = transform
        self.normalize = normalize
        self.path_to_data = path_to_data
        # self.filepaths = glob.glob(path_to_data + '/*.npz')
        self.inp_filepaths = inp_filepaths
        self.out_filepaths = out_filepaths
        self.z = torch.zeros((len(self.inp_filepaths), latent_dim, 2))
        # Ensure consistent ordering of paths

    def __getitem__(self, index):
        # inputs
        # Dictionary containing latitude, longitude and temperature
        inp = np.load(os.path.join(
            self.path_to_data, self.inp_filepaths[index]))
        latitude_in = inp["latitude"]  # Shape (num_lats,)
        longitude_in = inp["longitude"]  # Shape (num_lons,)
        temperature_in = inp["temperature"]  # Shape (num_lats, num_lons)
        if self.normalize:
            temperature_in = (temperature_in - T_MIN) / (T_MAX - T_MIN)
        # Create a grid of latitude and longitude values matching the shape
        # of the temperature grid
        longitude_grid_in, latitude_grid_in = np.meshgrid(
            longitude_in, latitude_in)
        coords_in = np.stack([latitude_grid_in, longitude_grid_in])
        # Shape (3, num_lats, num_lons)
        # data_tensor = np.stack([latitude_grid, longitude_grid, temperature])
        # data_tensor = torch.Tensor(data_tensor)
        coords_in = torch.Tensor(coords_in).permute(1, 2, 0)
        temperature_in = torch.Tensor(temperature_in).unsqueeze(-1)
        # Perform optional transform
        # self.transform:
        #    data_tensor_in = self.transform(data_tensor_in)

        out = np.load(os.path.join(
            self.path_to_data, self.out_filepaths[index]))
        latitude_out = out["latitude"]  # Shape (num_lats,)
        longitude_out = out["longitude"]  # Shape (num_lons,)
        temperature_out = out["temperature"]  # Shape (num_lats, num_lons)
        if self.normalize:
            temperature_out = (temperature_out - T_MIN) / (T_MAX - T_MIN)
        # Create a grid of latitude and longitude values matching the shape
        # of the temperature grid
        longitude_grid_out, latitude_grid_out = np.meshgrid(
            longitude_out, latitude_out)
        coords_out = np.stack([latitude_grid_out, longitude_grid_out])
        # Shape (3, num_lats, num_lons)
        # data_tensor = np.stack([latitude_grid, longitude_grid, temperature])
        # data_tensor = torch.Tensor(data_tensor)
        coords_out = torch.Tensor(coords_out).permute(1, 2, 0)
        temperature_out = torch.Tensor(temperature_out).unsqueeze(-1)
        # Perform optional transform

        coords = torch.cat(
            [coords_in.unsqueeze(-1), coords_out.unsqueeze(-1)], axis=-1)
        images = torch.cat(
            [temperature_in.unsqueeze(-1), temperature_out.unsqueeze(-1)], axis=-1
        )
        modulations = self.z[index, ...]

        # print('coords', coords.shape, images.shape)

        return (
            images,
            modulations,
            coords,
            index,
        )  # Label to ensure consistency with image datasets

    def __len__(self):
        return len(self.inp_filepaths)

    def __setitem__(self, z_values, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        z_values = z_values.clone()
        self.z[idx, ...] = z_values


class ERA5DatasetMemory(Dataset):
    """ERA5 temperature dataset.
    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """

    def __init__(
        self, path_to_data, filepaths, latent_dim, transform=None, normalize=True
    ):
        # self.path_to_data = path_to_data
        self.transform = transform
        self.normalize = normalize
        self.path_to_data = path_to_data
        self.filepaths = filepaths
        u_sample = np.load(os.path.join(path_to_data, filepaths[0]))[
            "temperature"]
        lat = np.load(os.path.join(path_to_data, filepaths[0]))["latitude"]
        long = np.load(os.path.join(path_to_data, filepaths[0]))["longitude"]
        lat_shape, long_shape = u_sample.shape[0], u_sample.shape[1]
        self.temp_min = T_MIN
        self.temp_max = T_MAX

        u = torch.zeros((len(filepaths), lat_shape, long_shape))

        for j in range(len(self.filepaths)):
            u_sample = np.load(os.path.join(path_to_data, filepaths[j]))[
                "temperature"]
            u[j] = (torch.Tensor(u_sample) - T_MIN) / (T_MAX - T_MIN)

        u = u.unsqueeze(-1)
        self.a = u[:-6]
        self.u = u[6:]
        self.z_a = torch.zeros((self.a.shape[0], latent_dim))
        self.z_u = torch.zeros((self.u.shape[0], latent_dim))
        long_grid, lat_grid = np.meshgrid(long, lat)
        self.coords = torch.Tensor(
            np.stack([lat_grid, long_grid])).permute(1, 2, 0)

        # Ensure consistent ordering of paths

    def __getitem__(self, index):
        a = self.a[index]
        u = self.u[index]
        z_a = self.z_a[index, ...]
        z_u = self.z_u[index, ...]

        return (
            a,
            u,
            z_a,
            z_u,
            self.coords.clone(),
            index,
        )  # Label to ensure consistency with image datasets

    def __len__(self):
        return self.a.shape[0]

    def __setitem__(self, z_a, z_u, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z_a[idx, ...] = z_a.cpu().clone()
        self.z_u[idx, ...] = z_u.cpu().clone()


class ERA5DatasetSimple(Dataset):
    """ERA5 temperature dataset.
    Args:
        path_to_data (string): Path to directory where data is stored.
        transform (torchvision.Transform): Optional transform to apply to data.
        normalize (bool): Whether to normalize data to lie in [0, 1]. Defaults
            to True.
    """

    def __init__(
        self, path_to_data, filepaths, latent_dim, transform=None, normalize=True
    ):
        # self.path_to_data = path_to_data
        self.transform = transform
        self.normalize = normalize
        self.path_to_data = path_to_data
        self.filepaths = filepaths
        u_sample = np.load(os.path.join(path_to_data, filepaths[0]))[
            "temperature"]
        lat = np.load(os.path.join(path_to_data, filepaths[0]))["latitude"]
        long = np.load(os.path.join(path_to_data, filepaths[0]))["longitude"]
        lat_shape, long_shape = u_sample.shape[0], u_sample.shape[1]
        self.temp_min = T_MIN
        self.temp_max = T_MAX

        u = torch.zeros((len(filepaths), lat_shape, long_shape))

        for j in range(len(self.filepaths)):
            u_sample = np.load(os.path.join(path_to_data, filepaths[j]))[
                "temperature"]
            u[j] = (torch.Tensor(u_sample) - T_MIN) / (T_MAX - T_MIN)

        self.u = u.unsqueeze(-1)
        self.z = torch.zeros((self.u.shape[0], latent_dim))
        long_grid, lat_grid = np.meshgrid(long, lat)
        self.coords = shape2spherical_coordinates((lat_shape, long_shape))

        # Ensure consistent ordering of paths

    def __getitem__(self, index):
        u = self.u[index]
        z = self.z[index]

        return (
            u,
            z,
            self.coords.clone(),
            index,
        )  # Label to ensure consistency with image datasets

    def __len__(self):
        return self.u.shape[0]

    def __setitem__(self, z, idx):
        # z_values = torch.tensor(np.array(z_values.clone()))
        self.z[idx, ...] = z.cpu().clone()


def shape2spherical_coordinates(spatial_shape):
    """Returns spherical coordinates on a uniform latitude and longitude grid.
    Args:
        spatial_shape (tuple of int): Tuple (num_lats, num_lons) containing
            number of latitudes and longitudes in grid.
    """
    num_lats, num_lons = spatial_shape
    # Uniformly spaced latitudes and longitudes corresponding to ERA5 grids
    latitude = torch.linspace(90.0, -90.0, num_lats)
    longitude = torch.linspace(0.0, 360.0 - (360.0 / num_lons), num_lons)
    # Create a grid of latitude and longitude values (num_lats, num_lons)
    longitude_grid, latitude_grid = torch.meshgrid(
        longitude, latitude, indexing="xy")
    # Create coordinate tensor
    # Spherical coordinates have 3 dimensions
    coordinates = torch.zeros(latitude_grid.shape + (3,))
    long_rad = deg_to_rad(longitude_grid)
    lat_rad = deg_to_rad(latitude_grid)
    coordinates[..., 0] = torch.cos(lat_rad) * torch.cos(long_rad)
    coordinates[..., 1] = torch.cos(lat_rad) * torch.sin(long_rad)
    coordinates[..., 2] = torch.sin(lat_rad)
    return coordinates


def deg_to_rad(degrees):
    return torch.pi * degrees / 180.0


def rad_to_deg(radians):
    return 180.0 * radians / torch.pi


class DataGenerator(torch.utils.data.Dataset):
    def __init__(
        self,
        ds,
        var_dict,
        lead_time,
        batch_size=32,
        shuffle=True,
        load=True,
        mean=None,
        std=None,
    ):
        """
        Data generator for WeatherBench data.
        Template from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        Args:
            ds: Dataset containing all variables
            var_dict: Dictionary of the form {'var': level}. Use None for level if data is of single level
            lead_time: Lead time in hours
            batch_size: Batch size
            shuffle: bool. If True, data is shuffled.
            load: bool. If True, datadet is loaded into RAM.
            mean: If None, compute mean from data.
            std: If None, compute standard deviation from data.
        """
        self.ds = ds
        self.var_dict = var_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.lead_time = lead_time

        data = []
        generic_level = xr.DataArray(
            [1], coords={"level": [1]}, dims=["level"])
        for var, levels in var_dict.items():
            try:
                data.append(ds[var].sel(levels))
            except ValueError:
                data.append(ds[var].expand_dims({"level": generic_level}, 1))

        self.data = xr.concat(data, "level").transpose(
            "time", "lat", "lon", "level")
        self.mean = (
            self.data.mean(("time", "lat", "lon")
                           ).compute() if mean is None else mean
        )
        self.std = (
            self.data.std("time").mean(
                ("lat", "lon")).compute() if std is None else std
        )
        # Normalize
        self.data = (self.data - self.mean) / self.std
        self.n_samples = self.data.isel(time=slice(0, -lead_time)).shape[0]
        self.init_time = self.data.isel(time=slice(None, -lead_time)).time
        self.valid_time = self.data.isel(time=slice(lead_time, None)).time

        self.on_epoch_end()

        # For some weird reason calling .load() earlier messes up the mean and std computations
        if load:
            print("Loading data into RAM")
            self.data.load()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return self.n_samples  # int(np.ceil(self.n_samples / self.batch_size))

    def __getitem__(self, i):
        "Generate one batch of data"
        # idxs = self.idxs[i * self.batch_size:(i + 1) * self.batch_size]
        X = self.data.isel(time=i).values
        y = self.data.isel(time=i + self.lead_time).values
        return X, y, i

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.idxs = np.arange(self.n_samples)
        if self.shuffle == True:
            np.random.shuffle(self.idxs)


def load_test_data(path, var, years=slice("2017", "2018")):
    """
    Load the test dataset. If z return z500, if t return t850.
    Args:
        path: Path to nc files
        var: variable. Geopotential = 'z', Temperature = 't'
        years: slice for time window
    Returns:
        dataset: Concatenated dataset for 2017 and 2018
    """
    ds = xr.open_mfdataset(f"{path}/*.nc", combine="by_coords")[var]
    if var in ["z", "t"]:
        if len(ds["level"].dims) > 0:
            try:
                ds = ds.sel(level=500 if var == "z" else 850).drop("level")
            except ValueError:
                ds = ds.drop("level")
        else:
            assert (
                ds["level"].values == 500 if var == "z" else ds["level"].values == 850
            )
    return ds.sel(time=years)


def compute_weighted_rmse(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    rmse = np.sqrt(((error) ** 2 * weights_lat).mean(mean_dims))
    return rmse


def compute_weighted_acc(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    clim = da_true.mean("time")
    try:
        t = np.intersect1d(da_fc.time, da_true.time)
        fa = da_fc.sel(time=t) - clim
    except AttributeError:
        t = da_true.time.values
        fa = da_fc - clim
    a = da_true.sel(time=t) - clim

    weights_lat = np.cos(np.deg2rad(da_fc.lat))
    weights_lat /= weights_lat.mean()
    w = weights_lat

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = np.sum(w * fa_prime * a_prime) / np.sqrt(
        np.sum(w * fa_prime**2) * np.sum(w * a_prime**2)
    )
    return acc


def compute_weighted_mae(da_fc, da_true, mean_dims=xr.ALL_DIMS):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = da_fc - da_true
    weights_lat = np.cos(np.deg2rad(error.lat))
    weights_lat /= weights_lat.mean()
    mae = (np.abs(error) * weights_lat).mean(mean_dims)
    return mae


def evaluate_iterative_forecast(da_fc, da_true, func, mean_dims=xr.ALL_DIMS):
    """
    Compute iterative score (given by func) with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Iterative Forecast. Time coordinate must be initialization time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        score: Latitude weighted score
    """
    rmses = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        fc["time"] = fc.time + np.timedelta64(int(f), "h")
        rmses.append(func(fc, da_true, mean_dims))
    return xr.concat(rmses, "lead_time")
