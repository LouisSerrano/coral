from pathlib import Path
import torch.nn as nn
import torch_geometric.nn as nng
import numpy as np
import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Identity
from torch_geometric.nn import Linear
from torch_geometric.nn.models import GraphSAGE, GraphUNet
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.data import Data
from equations.PDEs import *
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm

from coral.mlp import MLP

def create_model_instance(cfg, input_dim=1, output_dim=1, device="cuda"):
    device = torch.device(device)
    if cfg.model.model_type == "mlp":
        model = MLP(input_dim=input_dim,
                    hidden_dim=cfg.model.width,
                    output_dim=output_dim,
                    depth=cfg.model.depth,
                    dropout=cfg.model.dropout,
                    activation=cfg.model.activation).to(device)
    elif cfg.model.model_type == "sage":
        model = GraphSAGE(in_channels=input_dim,
                        hidden_channels=cfg.model.width,
                        num_layers=cfg.model.depth,
                        out_channels=output_dim,
                        dropout=cfg.model.dropout, 
                        act=cfg.model.activation,
                        act_first=False, 
                        norm=cfg.model.norm).to(device)
    elif cfg.model.model_type == "gunet":
        model = GraphUNet(in_channels=input_dim,
                        hidden_channels=cfg.model.width,
                        out_channels=output_dim,
                        depth=cfg.model.depth,
                        pool_ratios=cfg.model.pool_ratios,
                        act=cfg.model.activation).to(device)
    elif cfg.model.model_type == "mppde":
        pos_dim = 2
        model = MP_PDE_Solver(pos_dim=pos_dim,
                 input_dim=input_dim - pos_dim,
                 output_dim=output_dim,
                 time_window=1,
                 hidden_features=cfg.model.width,
                 hidden_layer=cfg.model.depth).to(device)
    else:
        raise NotImplementedError(f"No corresponding class for {cfg.model.model_type}")
    return model

def load_model(
    run_dir, run_name, input_dim=1, output_dim=1, device="cuda"
):
    model_train = torch.load(run_dir / f"{run_name}.pt")

    model_state_dict = model_train["model"]
    cfg = model_train["cfg"]

    model = create_model_instance(cfg, input_dim, output_dim, device)
    model.load_state_dict(model_state_dict)
    model.eval()

    return model

class MLPBatch(torch.nn.Module):
    r"""A multi-layer perception (MLP) model.
    Args:
        channel_list (List[int]): List of input, intermediate and output
            channels. :obj:`len(channel_list) - 1` denotes the number of layers
            of the MLP.
        dropout (float, optional): Dropout probability of each hidden
            embedding. (default: :obj:`0.`)
        batch_norm (bool, optional): If set to :obj:`False`, will not make use
            of batch normalization. (default: :obj:`True`)
        relu_first (bool, optional): If set to :obj:`True`, ReLU activation is
            applied before batch normalization. (default: :obj:`False`)
    """
    def __init__(self, channel_list, dropout = 0.,
                 batch_norm = True, relu_first = False):
        super().__init__()
        assert len(channel_list) >= 2
        self.channel_list = channel_list
        self.dropout = dropout
        self.relu_first = relu_first

        self.lins = torch.nn.ModuleList()
        for dims in zip(self.channel_list[:-1], self.channel_list[1:]):
            self.lins.append(Linear(*dims))

        self.norms = torch.nn.ModuleList()
        for dim in zip(self.channel_list[1:-1]):
            self.norms.append(BatchNorm1d(dim, track_running_stats = False) if batch_norm else Identity())

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for norm in self.norms:
            if hasattr(norm, 'reset_parameters'):
                norm.reset_parameters()


    def forward(self, x: Tensor) -> Tensor:
        """"""
        x = self.lins[0](x)
        for lin, norm in zip(self.lins[1:], self.norms):
            if self.relu_first:
                x = x.relu_()
            x = norm(x)
            if not self.relu_first:
                x = x.relu_()
            x = F.dropout(x, p = self.dropout, training = self.training)
            x = lin.forward(x)
        return x


class GraphSAGE2(nn.Module):
    def __init__(self, input_dim, output_dim, width=256, depth=3, batch_norm=True, width_encoder=128, width_decoder=128, depth_encoder=2, depth_decoder=2):
        super(GraphSAGE2, self).__init__()

        self.nb_hidden_layers = depth
        self.size_hidden_layers = width
        self.bn_bool = batch_norm
        self.activation = nn.ReLU()

        self.encoder = MLPBatch([width_encoder]*depth_encoder)
        self.decoder = MLPBatch([width_decoder]*depth_decoder)

        self.in_layer = nng.SAGEConv(
            in_channels=input_dim,
            out_channels=self.size_hidden_layers
        )

        self.hidden_layers = nn.ModuleList()
        for n in range(self.nb_hidden_layers - 1):
            self.hidden_layers.append(nng.SAGEConv(
                in_channels = self.size_hidden_layers,
                out_channels = self.size_hidden_layers
            ))

        self.out_layer = nng.SAGEConv(
                in_channels = self.size_hidden_layers,
                out_channels = output_dim
            )

        if self.bn_bool:
            self.bn = nn.ModuleList()
            for n in range(self.nb_hidden_layers):
                self.bn.append(nn.BatchNorm1d(self.size_hidden_layers, track_running_stats = False))

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        z = self.encoder(z)
        
        z = self.in_layer(z, edge_index)
        if self.bn_bool:
            z = self.bn[0](z)
        z = self.activation(z)

        for n in range(self.nb_hidden_layers - 1):
            z = self.hidden_layers[n](z, edge_index)
            if self.bn_bool:
                z = self.bn[n + 1](z)
            z = self.activation(z)

        z = self.out_layer(z, edge_index)

        z = self.decoder(z)

        return z

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class GNN_Layer(MessagePassing):
    """
    Message passing layer
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 hidden_features: int,
                 time_window: int,
                 input_dim: int = 3,
                 pos_dim: int = 2):
        """
        Initialize message passing layers
        Args:
            in_features (int): number of node input features
            out_features (int): number of node output features
            hidden_features (int): number of hidden features
            time_window (int): number of input/output timesteps (temporal bundling)
            n_variables (int): number of equation specific parameters used in the solver
        """
        super(GNN_Layer, self).__init__(node_dim=-2, aggr='mean')
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + input_dim + pos_dim, hidden_features),
                                           Swish()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           Swish()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features, hidden_features),
                                          Swish()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          Swish()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, edge_index, batch):
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos)
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j):
        """
        Message update following formula 8 of the paper
        """
        message = self.message_net_1(torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j), dim=-1))
        message = self.message_net_2(message)
        return message

    def update(self, message, x):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message), dim=-1))
        update = self.update_net_2(update)
        if self.in_features == self.out_features:
            return x + update
        else:
            return update


class MP_PDE_Solver(torch.nn.Module):
    """
    MP-PDE solver class
    """
    def __init__(self,
                 pos_dim: int = 2,
                 input_dim: int = 3,
                 output_dim: int = 3,
                 time_window: int = 1,
                 hidden_features: int = 64,
                 hidden_layer: int = 6,
    ):
        """
        Initialize MP-PDE solver class.
        It contains 6 MP-PDE layers with skip connections
        The input graph to the forward pass has the shape [batch*n_nodes, time_window].
        The output graph has the shape [batch*n_nodes, time_window].
        Args:
            pde (PDE): PDE at hand
            time_window (int): number of input/output timesteps (temporal bundling)
            hidden features (int): number of hidden features
            hidden_layer (int): number of hidden layers
            eq_variables (dict): dictionary of equation specific parameters
        """
        super(MP_PDE_Solver, self).__init__()
        # 1D decoder CNN is so far designed time_window = [20,25,50]
        self.pos_dim = pos_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            input_dim=self.input_dim,
            pos_dim=self.pos_dim
            #n_variables=len(self.eq_variables) + 1  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         input_dim=self.input_dim,
                                         pos_dim=self.pos_dim
                                         #n_variables=len(self.eq_variables) + 1
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window * self.input_dim + self.pos_dim , self.hidden_features), # + len(self.eq_variables)
            Swish(),
            nn.Linear(self.hidden_features, self.hidden_features),
            Swish()
        )


        # Decoder CNN, maps to different outputs (temporal bundling)
        if(self.time_window==20):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 15, stride=4),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1)
                                            )
        if (self.time_window == 25):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 16, stride=3),
                                            Swish(),
                                            nn.Conv1d(8, 1, 14, stride=1)
                                            )
        if(self.time_window==50):
            self.output_mlp = nn.Sequential(nn.Conv1d(1, 8, 12, stride=2),
                                            Swish(),
                                            nn.Conv1d(8, 1, 10, stride=1))
        if (self.time_window == 1):
            self.output_mlp = nn.Sequential(nn.Linear(hidden_features, 64), Swish(), nn.Linear(64, self.output_dim*self.time_window))
        

    def __repr__(self):
        return f'GNN'

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass of MP-PDE solver class.
        The input graph has the shape [batch*n_nodes, time_window].
        The output tensor has the shape [batch*n_nodes, time_window].
        Args:
            data (Data): Pytorch Geometric data graph
        Returns:
            torch.Tensor: data output
        """
        u = data.input[..., self.pos_dim:]
        pos = data.input[..., :self.pos_dim]
        # Encode and normalize coordinate information
        #pos = data.pos[..., 0] # we take the pos from the initial condition
        #pos_x = pos[:, 1][:, None] / self.pde.L
        #pos_t = pos[:, 0][:, None] / self.pde.tmax
        edge_index = data.edge_index
        #print('edge_index', edge_index.shape, edge_index.dtype)
        batch = data.batch

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments
        #variables = pos_t    # time is treated as equation variable
        #if "alpha" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.alpha / self.eq_variables["alpha"]), -1)
        #if "beta" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.beta / self.eq_variables["beta"]), -1)
        #if "gamma" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.gamma / self.eq_variables["gamma"]), -1)
        #if "bc_left" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.bc_left), -1)
        #if "bc_right" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.bc_right), -1)
        #if "c" in self.eq_variables.keys():
        #    variables = torch.cat((variables, data.c / self.eq_variables["c"]), -1)

        # Encoder and processor (message passing)
        #node_input = torch.cat((u, pos_x, variables), -1)
        node_input = torch.cat((u, pos), -1)

        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos, edge_index, batch)

        # Decoder (formula 10 in the paper)
        #dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        #dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h[:, None]).squeeze(1)
        #out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff
        out = u + diff

        return out