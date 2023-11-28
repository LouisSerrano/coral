import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius_graph
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing, global_mean_pool, InstanceNorm, avg_pool_x, BatchNorm
from mppde.baseline_coral.utils_coral import PDE_CORAL

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
                 n_variables: int):
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

        self.message_net_1 = nn.Sequential(nn.Linear(2 * in_features + n_variables + 1, hidden_features),
                                           nn.ReLU()
                                           )
        self.message_net_2 = nn.Sequential(nn.Linear(hidden_features, hidden_features),
                                           nn.ReLU()
                                           )
        self.update_net_1 = nn.Sequential(nn.Linear(in_features + hidden_features + 1, hidden_features),
                                          nn.ReLU()
                                          )
        self.update_net_2 = nn.Sequential(nn.Linear(hidden_features, out_features),
                                          nn.ReLU()
                                          )
        self.norm = InstanceNorm(hidden_features)

    def forward(self, x, u, pos, variables, edge_index, batch): 
        """
        Propagate messages along edges
        """
        x = self.propagate(edge_index, x=x, u=u, pos=pos, variables=variables) 
        x = self.norm(x, batch)
        return x

    def message(self, x_i, x_j, u_i, u_j, pos_i, pos_j, variables_i):
        """
        Message update following formula 8 of the paper
        """
        message = torch.cat((x_i, x_j, u_i - u_j, pos_i - pos_j, variables_i), dim=-1)
        message = self.message_net_1(message)
        message = self.message_net_2(message)
        return message

    def update(self, message, x, variables):
        """
        Node update following formula 9 of the paper
        """
        update = self.update_net_1(torch.cat((x, message, variables), dim=-1))
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
                 pde: PDE_CORAL,
                 time_window: int = 25,
                 hidden_features: int = 128,
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
        #assert(time_window == 25 or time_window == 20 or time_window == 50)
        self.pde = pde
        self.out_features = time_window
        self.hidden_features = hidden_features
        self.hidden_layer = hidden_layer
        self.time_window = time_window

        self.gnn_layers = torch.nn.ModuleList(modules=(GNN_Layer(
            in_features=self.hidden_features,
            hidden_features=self.hidden_features,
            out_features=self.hidden_features,
            time_window=self.time_window,
            n_variables= pde.pos_dim + 1,  # variables = eq_variables + time
        ) for _ in range(self.hidden_layer - 1)))

        # The last message passing last layer has a fixed output size to make the use of the decoder 1D-CNN easier
        self.gnn_layers.append(GNN_Layer(in_features=self.hidden_features,
                                         hidden_features=self.hidden_features,
                                         out_features=self.hidden_features,
                                         time_window=self.time_window,
                                         n_variables=pde.pos_dim + 1,
                                        )
                               )

        self.embedding_mlp = nn.Sequential(
            nn.Linear(self.time_window + self.pde.pos_dim + 1, self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features, self.hidden_features),
            nn.ReLU()
        )
        self.output_mlp = nn.Sequential(
            nn.Conv1d(1, 8, 15, stride=4),
            nn.ReLU(),
            nn.Conv1d(8, 4, 10, stride=2),
            nn.ReLU(),
            nn.Conv1d(4, 1, 10, stride=1),
        )

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
        u = data.x
        # Encode and normalize coordinate information
        pos = data.pos
        pos_t = pos[:, 0:1]
        variables = pos_t    # time is treated as equation variable

        pos = pos[:, 1:]
        
        edge_index = data.edge_index
        batch = data.batch

        # Encode equation specific parameters
        # alpha, beta, gamma are used in E1,E2,E3 experiments
        # bc_left, bc_right, c are used in WE1, WE2, WE3 experiments

        # Encoder and processor (message passing)

        if self.pde.pos_dim == 3:
            if self.pde.input_dim == 2:
                pos = pos.repeat(2, 1)
                variables = variables.repeat(2, 1)
                batch = batch.repeat(2)
            node_input = torch.cat((u, pos[:, 0:1], pos[:, 1:2], pos[:, 2:3], variables), -1) # 
        if self.pde.pos_dim == 2:
            if self.pde.input_dim == 2:
                pos = pos.repeat(2, 1)
                variables = variables.repeat(2, 1)
                batch = batch.repeat(2)
            node_input = torch.cat((u, pos[:, 0:1], pos[: , 1:2], variables), -1) #
        if self.pde.pos_dim == 1:
            if self.pde.input_dim == 2:
                pos = pos.repeat(2, 1)
                variables = variables.repeat(2, 1)
                batch = batch.repeat(2)
            node_input = torch.cat((u, pos[:, 0:1], variables), -1) # 
        h = self.embedding_mlp(node_input)
        for i in range(self.hidden_layer):
            h = self.gnn_layers[i](h, u, pos, variables, edge_index, batch) # 

        # Decoder (formula 10 in the paper)
        dt = (torch.ones(1, self.time_window) * self.pde.dt).to(h.device)
        dt = torch.cumsum(dt, dim=1)
        # [batch*n_nodes, hidden_dim] -> 1DCNN([batch*n_nodes, 1, hidden_dim]) -> [batch*n_nodes, time_window]
        diff = self.output_mlp(h.unsqueeze(1)).squeeze(1)
        out = u[:, -1].repeat(self.time_window, 1).transpose(0, 1) + dt * diff
        return out