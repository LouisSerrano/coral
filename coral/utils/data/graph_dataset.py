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
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import json
import functools
import tensorflow as tf
from coral.mlp import MLP

KEY_TO_INDEX = {
    "cylinder-flow": {"pressure": 0, "vx": 1, "vy": 2},
    "airfoil-flow": {"pressure": 0, "density": 1, "vx": 2, "vy": 3},
}

KEY_TO_STATS = {
    "airfoil-flow": {
        "pressure": {"mean": 100408.1953, "std": 9741.1650},
        "density": {"mean": 1.2130, "std": 0.0915},
        "vx": {"mean": 167.4502, "std": 71.4523},
        "vy": {"mean": -1.1697, "std": 50.0617},
        "pos": {"min": -20, "max": 20},
    }
}

def _parse(proto, meta):
    """Parses a trajectory from tf.Example."""
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def load_dataset(path, split):
    """Load dataset."""
    with open(os.path.join(path, "meta.json"), "r") as fp:
        meta = json.loads(fp.read())
    ds = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)
    return ds


class CylinderFlowDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        path="/data/serrano/inr_domain_decomposition/cylinder_flow",
        split="train",
        latent_dim=128,
        noise=0.02,
        task="static",
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        super().__init__(None, None, None)
        self.path = path
        self.split = split
        self.latent_dim = latent_dim
        self.T = 2
        if task == "static":
            self.dataset = self.load_static_data()
        else:
            raise NotImplementedError

    def load_static_data(self):
        filename = Path(self.path) / f"static_{self.split}.h5"
        self._indices = []
        try:
            f = h5py.File(filename, "r")
            f.close()

        except:
            ds = load_dataset(self.path, self.split)
            f = h5py.File(filename, "w")
            # we take the first and last timestamps
            for index, d in enumerate(ds):
                # (t, n, c) -> (n, c, t)
                pos = d["mesh_pos"].numpy()[[0, -1]].transpose(1, 2, 0)
                node_type = d["node_type"].numpy()[[0, -1]].transpose(1, 2, 0)
                velocity = d["velocity"].numpy()[[0, -1]].transpose(1, 2, 0)
                cells = d["cells"].numpy()[[0, -1]].transpose(1, 2, 0)
                pressure = d["pressure"].numpy()[[0, -1]].transpose(1, 2, 0)

                data = ("pos", "node_type", "velocity", "cells", "pressure")
                # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
                g = f.create_group(str(index))
                for k in data:
                    g[k] = eval(k)
            f.close()

        dataset = {}
        with h5py.File(filename, "r") as f:
            for index in range(len(f)):
                self._indices.append(index)
                v = torch.from_numpy(f[f"{index}"]["velocity"][()])
                p = torch.from_numpy(f[f"{index}"]["pressure"][()])
                pos = torch.from_numpy(f[f"{index}"]["pos"][()])
                node_type = torch.from_numpy(f[f"{index}"]["node_type"][()])
                cells = torch.from_numpy(f[f"{index}"]["cells"][()])
                tmp = Data(pos=pos)
                tmp.v = v
                tmp.p = p
                tmp.node_type = node_type
                tmp.cells = cells
                tmp.z_v = torch.zeros(1, self.latent_dim, 2)
                tmp.z_vx = torch.zeros(1, self.latent_dim, 2)
                tmp.z_vy = torch.zeros(1, self.latent_dim, 2)
                tmp.z_p = torch.zeros(1, self.latent_dim, 2)
                tmp.z_geo = torch.zeros(1, self.latent_dim, 2)
                dataset[f"{index}"] = tmp

        return dataset

    def len(self):
        return len(self.dataset)

    def get(self, key):
        #  print('after reading', time() - t0)

        # graph = Data(
        #    v0=v0 + torch.randn_like(v0) * self.noise / self.v_sigma, edge_index=None
        # )  # test with only one value

        graph = self.dataset[f"{key}"].clone()
        graph.z_input = torch.cat(
            [graph.z_p[..., 0], graph.z_vx[..., 0], graph.z_vy[..., 0]], axis=-1
        )
        graph.z_output = torch.cat(
            [graph.z_p[..., 1], graph.z_vx[..., 1], graph.z_vy[..., 1]], axis=-1
        )
        graph.input = torch.cat([graph.pos[..., 0], graph.p[..., 0], graph.v[..., 0]], axis=-1).float()
        graph.images = torch.cat([graph.p[..., 1], graph.v[..., 1]], axis=-1).float()

        return (graph, key)

    def set_code(self, z_values, key):
        z_values = z_values.cpu().detach()
        for index in range(self.len()):
            if key == "vx":
                self.dataset[f"{index}"].z_vx = z_values[index].unsqueeze(0)

            elif key == "vy":
                self.dataset[f"{index}"].z_vy = z_values[index].unsqueeze(0)

            elif key == "velocity":
                self.dataset[f"{index}"].z_v = z_values[index].unsqueeze(0)

            elif key == "pressure":
                self.dataset[f"{index}"].z_p = z_values[index].unsqueeze(0)

            elif key == "geometry":
                # we only need one sample for the geometry
                self.dataset[f"{index}"].z_geo = z_values[index].unsqueeze(0)


class AirfoilFlowDataset(Dataset):
    """Custom dataset for encoding task. Contains the values, the codes, and the coordinates for input and output."""

    def __init__(
        self,
        path="/data/serrano/inr_domain_decomposition/airfoil",
        split="train",
        latent_dim=128,
        noise=0.02,
        task="static",
    ):
        """
        Args:
            x (torch.Tensor): Dataset input values
            y (torch.Tensor): Dataset output values
            grid_x (torch.Tensor): input coordinates
            grid_y (torch.Tensor): output coordinates
            latent_dim (int, optional): Latent dimension of the code. Defaults to 256.
        """
        super().__init__(None, None, None)
        self.path = path
        self.split = split
        self.latent_dim = latent_dim
        self.T = 2
        if task == "static":
            self.dataset = self.load_static_data()
        else:
            raise NotImplementedError

    def load_static_data(self):
        filename = Path(self.path) / f"static_{self.split}.h5"
        self._indices = []
        try:
            f = h5py.File(filename, "r")
            f.close()

        except:
            ds = load_dataset(self.path, self.split)
            f = h5py.File(filename, "w")
            # we take the first and last timestamps
            for index, d in enumerate(ds):
                # (t, n, c) -> (n, c, t)
                pos = d["mesh_pos"].numpy()[[0, -1]].transpose(1, 2, 0)
                node_type = d["node_type"].numpy()[[0, -1]].transpose(1, 2, 0)
                velocity = d["velocity"].numpy()[[0, -1]].transpose(1, 2, 0)
                cells = d["cells"].numpy()[[0, -1]].transpose(1, 2, 0)
                density = d["density"].numpy()[[0, -1]].transpose(1, 2, 0)
                pressure = d["pressure"].numpy()[[0, -1]].transpose(1, 2, 0)
                data = ("pos", "node_type", "velocity", "cells", "density", "pressure")
                # d = f.create_dataset(str(index), (len(data), ), dtype=pos.dtype)
                g = f.create_group(str(index))
                for k in data:
                    g[k] = eval(k)
            f.close()

        dataset = {}
        with h5py.File(filename, "r") as f:
            for index in range(len(f)):
                self._indices.append(index)
                v = torch.from_numpy(f[f"{index}"]["velocity"][()])
                p = torch.from_numpy(f[f"{index}"]["pressure"][()])
                rho = torch.from_numpy(f[f"{index}"]["density"][()])
                pos = torch.from_numpy(f[f"{index}"]["pos"][()])
                node_type = torch.from_numpy(f[f"{index}"]["node_type"][()])
                cells = torch.from_numpy(f[f"{index}"]["cells"][()])
                tmp = Data(pos=pos)
                tmp.v = v
                tmp.p = p
                tmp.rho = rho
                tmp.node_type = node_type
                tmp.cells = cells
                tmp.z_v = torch.zeros(1, self.latent_dim, 2)
                tmp.z_vx = torch.zeros(1, self.latent_dim, 2)
                tmp.z_vy = torch.zeros(1, self.latent_dim, 2)
                tmp.z_rho = torch.zeros(1, self.latent_dim, 2)
                tmp.z_p = torch.zeros(1, self.latent_dim, 2)
                tmp.z_geo = torch.zeros(1, self.latent_dim, 2)
                dataset[f"{index}"] = tmp

        return dataset

    def len(self):
        return len(self.dataset)

    def get(self, key):
        graph = self.dataset[f"{key}"].clone()
        graph = self.normalize(graph)

        graph.z_input = torch.cat(
            [
                graph.z_p[..., 0],
                graph.z_rho[..., 0],
                graph.z_vx[..., 0],
                graph.z_vy[..., 0],
            ],
            axis=-1,
        )
        graph.z_output = torch.cat(
            [
                graph.z_p[..., 1],
                graph.z_rho[..., 1],
                graph.z_vx[..., 1],
                graph.z_vy[..., 1],
            ],
            axis=-1,
        )
        graph.input = torch.cat([graph.pos[..., 0], graph.p[..., 0], graph.rho[..., 0], graph.v[..., 0]], axis=-1).float()
        graph.images = torch.cat(
            [graph.p[..., 1], graph.rho[..., 1], graph.v[..., 1]], axis=-1
        ).float()

        return (graph, key)

    def normalize(self, graph):
        tmp_dic = KEY_TO_STATS["airfoil-flow"]
        p_mu, p_sigma = tmp_dic["pressure"]['mean'], tmp_dic["pressure"]['std'] 
        rho_mu, rho_sigma = tmp_dic["density"]['mean'], tmp_dic["density"]['std'] 
        vx_mu, vx_sigma = tmp_dic["vx"]['mean'], tmp_dic["vx"]['std'] 
        vy_mu, vy_sigma = tmp_dic["vy"]['mean'], tmp_dic["vy"]['std'] 
        pos_max = tmp_dic["pos"]['max']

        graph.p = (graph.p - p_mu) / p_sigma
        graph.rho = (graph.rho - rho_mu) / rho_sigma
        graph.v[:,0,:] = (graph.v[:, 0, :] - vx_mu) / vx_sigma
        graph.v[:,1,:] = (graph.v[:, 1, :] - vy_mu) / vy_sigma
        graph.pos = graph.pos / pos_max

        return graph

    def set_code(self, z_values, key):
        z_values = z_values.cpu().detach()
        for index in range(self.len()):
            if key == "vx":
                self.dataset[f"{index}"].z_vx = z_values[index].unsqueeze(0)

            elif key == "vy":
                self.dataset[f"{index}"].z_vy = z_values[index].unsqueeze(0)

            elif key == "velocity":
                self.dataset[f"{index}"].z_v = z_values[index].unsqueeze(0)

            elif key == "pressure":
                self.dataset[f"{index}"].z_p = z_values[index].unsqueeze(0)

            elif key == "density":
                self.dataset[f"{index}"].z_rho = z_values[index].unsqueeze(0)

            elif key == "geometry":
                # we only need one sample for the geometry
                self.dataset[f"{index}"].z_geo = z_values[index].unsqueeze(0)


if __name__ == "__main__":
    dataset_name = "airfoil-flow"

    if dataset_name == "cylinder-flow":
        path = Path("/data/serrano/inr_domain_decomposition/cylinder_flow")
        trainset = CylinderFlowDataset(
            path=path, split="test", latent_dim=128, noise=0.02, task="static"
        )
        #print("len", len(trainset))
        train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True)

        for graph, idx in train_loader:
            break
        #print("graph", graph.v.shape)
    else:
        model = MLP(input_dim=6,
                hidden_dim=256,
                output_dim=4,
                depth=3,
                dropout=0).cuda()
        path = Path("/data/serrano/inr_domain_decomposition/airfoil")
        trainset = AirfoilFlowDataset(
            path=path, split="train", latent_dim=128, noise=0.02, task="static"
        )
        #print("len", len(trainset))
        train_loader = DataLoader(dataset=trainset, batch_size=32, shuffle=True)

        vx_all = []
        vy_all = []
        p_all = []
        rho_all = []
        posx_all = []
        posy_all = []
        
        for graph, idx in train_loader:
            break
        graph = graph.cuda()
        print(graph.input.dtype)
        pred = model(graph.input)
        print('pred', pred.dtype)


        for graph, idx in train_loader:
            print('toto len', len(graph))
            print('toto')
            print('toto')
            exit()

            print("v", graph.v.mean(), graph.v.std())
            print("p", graph.p.mean(), graph.p.std())
            print("rho", graph.rho.mean(), graph.rho.std())
            print(
                "pos",
                graph.pos.mean(),
                graph.pos.std(),
                graph.pos.max(),
                graph.pos.min(),
            )
            print("posx", graph.pos[:, 0, :].max(), graph.pos[:, 0, :].min())
            print("posy", graph.pos[:, 1, :].max(), graph.pos[:, 1, :].min())
            vx_all.append(graph.v[:, 0, :].flatten())
            vy_all.append(graph.v[:, 1, :].flatten())
            p_all.append(graph.p.flatten())
            rho_all.append(graph.rho.flatten())
            posx_all.append(graph.pos[:, 0, :].flatten())
            posy_all.append(graph.pos[:, 1, :].flatten())

        vx_all = torch.cat(vx_all)
        vy_all = torch.cat(vy_all)
        p_all = torch.cat(p_all)
        rho_all = torch.cat(rho_all)
        posx_all = torch.cat(posx_all)
        posy_all = torch.cat(posy_all)

        print("staaaaats")
        print("vx", vx_all.mean(), vx_all.std())
        print("vx", vy_all.mean(), vy_all.std())
        print("p", p_all.mean(), p_all.std())
        print("rho", rho_all.mean(), rho_all.std())
        print("posx", posx_all.min(), posx_all.max())
        print("posy", posy_all.min(), posy_all.max())
