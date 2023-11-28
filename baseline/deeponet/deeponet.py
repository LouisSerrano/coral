import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

class Swish(nn.Module):
    """
    Swish activation function
    """
    def __init__(self, beta=1):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta*x)


class Module(torch.nn.Module):
    '''Standard module format.'''
    def __init__(self):
        super(Module, self).__init__()
        self.activation = None
        self.initializer = None
        self.__device = None
        self.__dtype = None

    @property
    def device(self):
        return self.__device

    @property
    def dtype(self):
        return self.__dtype

    @device.setter
    def device(self, d):
        if d == 'cpu':
            self.cpu()
        elif d == 'gpu':
            self.cuda()
        else:
            raise ValueError
        self.__device = d

    @dtype.setter
    def dtype(self, d):
        if d == 'float':
            self.to(torch.float)
        elif d == 'double':
            self.to(torch.double)
        else:
            raise ValueError
        self.__dtype = d

    @property
    def Device(self):
        if self.__device == 'cpu':
            return torch.device('cpu')
        elif self.__device == 'gpu':
            return torch.device('cuda')

    @property
    def Dtype(self):
        if self.__dtype == 'float':
            return torch.float32
        elif self.__dtype == 'double':
            return torch.float64

    @property
    def act(self):
        if self.activation == 'sigmoid':
            return torch.sigmoid
        elif self.activation == 'relu':
            return torch.relu
        elif self.activation == 'tanh':
            return torch.tanh
        elif self.activation == 'elu':
            return torch.elu
        elif self.activation == 'swish':
            return Swish()
        else:
            raise NotImplementedError

    @property
    def Act(self):
        if self.activation == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.activation == 'relu':
            return torch.nn.ReLU()
        elif self.activation == 'tanh':
            return torch.nn.Tanh()
        elif self.activation == 'elu':
            return torch.nn.ELU()
        elif self.activation == 'swish':
            return Swish()
        else:
            raise NotImplementedError

    @property
    def weight_init_(self):
        if self.initializer == 'He normal':
            return torch.nn.init.kaiming_normal_
        elif self.initializer == 'He uniform':
            return torch.nn.init.kaiming_uniform_
        elif self.initializer == 'Glorot normal':
            return torch.nn.init.xavier_normal_
        elif self.initializer == 'Glorot uniform':
            return torch.nn.init.xavier_uniform_
        elif self.initializer == 'orthogonal':
            return torch.nn.init.orthogonal_
        elif self.initializer == 'default':
            if self.activation == 'relu':
                return torch.nn.init.kaiming_normal_
            elif self.activation == 'tanh':
                return torch.nn.init.orthogonal_
            else:
                return lambda x: None
        else:
            raise NotImplementedError


class StructureNN(Module):
    '''Structure-oriented neural network used as a general map based on designing architecture.'''
    def __init__(self):
        super(StructureNN, self).__init__()

    def predict(self, x, returnnp=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=self.Dtype, device=self.Device)
        return self(x).cpu().detach().numpy() if returnnp else self(x)


class DeepONet(StructureNN):
    '''
    Deep operator network.
    Input: [batch size, branch_dim + trunk_dim]
    Output: [batch size, 1]
    '''
    def __init__(self, branch_dim, trunk_dim, branch_depth=2, trunk_depth=3, width=50, activation='relu', logger=None, input_dataset="navier"):
        super(DeepONet, self).__init__()
        self.branch_dim = branch_dim
        self.trunk_dim = trunk_dim
        self.branch_depth = branch_depth
        self.trunk_depth = trunk_depth
        self.width = width
        self.activation = activation
        self.modus = self.__init_modules()
        self.params = self.__init_params()
        self.logger = logger
        self.is_sw = (input_dataset == "shallow-water-dino")

    def forward(self, x_branch, x_trunk):
        # X_branch = B, XY
        # X_trunk = B, XY, grid
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        # x trunk  = torch.Size([6, 4096, 250])
    
        for i in range(1, self.branch_depth):
            #  [10, 1638, 2]
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        x_branch = x_branch.unsqueeze(dim=1).expand(x_trunk.shape)
        # X_branch = # [6, 4096, 250]

        if self.is_sw: # TODO : multichannels
            output1 = torch.sum(x_branch[:, :, :self.width // 2] * x_trunk[:, :, :self.width // 2], dim=-1, keepdim=True) + self.params['bias']
            output2 = torch.sum(x_branch[:, :, self.width // 2:] * x_trunk[:, :, self.width // 2:], dim=-1, keepdim=True) + self.params['bias2']
            output = torch.cat((output1, output2), dim=2)
        else:
            output = torch.sum(x_branch * x_trunk, dim=-1, keepdim=True) + self.params['bias'] # TODO multichannels

        return output

    def trunk_forward(self, x_trunk):
        for i in range(1, self.trunk_depth):
            x_trunk = self.modus['TrActM{}'.format(i)](self.modus['TrLinM{}'.format(i)](x_trunk))
        return x_trunk

    def branch_forward(self, x_branch):
        for i in range(1, self.branch_depth):
            x_branch = self.modus['BrActM{}'.format(i)](self.modus['BrLinM{}'.format(i)](x_branch))
        x_branch = self.modus['BrLinM{}'.format(self.branch_depth)](x_branch)
        return x_branch

    def __init_modules(self):
        modules = nn.ModuleDict()
        if self.branch_depth > 1:
            modules['BrLinM1'] = nn.Linear(self.branch_dim, self.width)
            modules['BrActM1'] = self.Act
            for i in range(2, self.branch_depth):
                modules['BrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
                modules['BrActM{}'.format(i)] = self.Act
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.width, self.width)
        else:
            modules['BrLinM{}'.format(self.branch_depth)] = nn.Linear(self.branch_dim, self.width)

        modules['TrLinM1'] = nn.Linear(self.trunk_dim, self.width)
        modules['TrActM1'] = self.Act
        for i in range(2, self.trunk_depth):
            modules['TrLinM{}'.format(i)] = nn.Linear(self.width, self.width)
            modules['TrActM{}'.format(i)] = self.Act
        return modules

    def __init_params(self):
        params = nn.ParameterDict()
        params['bias'] = nn.Parameter(torch.zeros([1]))
        params['bias2'] = nn.Parameter(torch.zeros([1]))
        return params


def AR_forward(deeponet, t, coords, ground_truth, logger=None, is_test=False):
    # TODO is_test à tester
    
    if is_test:
        x_trunk_in_s = coords.reshape(*coords.shape[:1], -1, *coords.shape[-1:])
    else:
        x_trunk_in_s = coords[:, :, :]
    x_branch = ground_truth[:, :1, :, :].squeeze()  # [32, 10, 1992, 1] vs [32, 10, 1992, 2]
    # CI, on prend que le 1er pas de temps et on squeeze = B, XY, C
    if len(x_branch.shape) == 3:
        x_branch = x_branch.flatten(start_dim=1)
    for i, t in enumerate(t):
        # x_branch is B, XY
        # x_trink_in_s is B, T, XY, grid

        # X_trunk devrait etre B, XY, grid et X_branch devrait etre B, XY
        im = deeponet(x_branch, x_trunk_in_s).unsqueeze(dim=1)

        if i == 0: # manage initial condition
            if is_test:
                model_output = ground_truth.reshape(*ground_truth.shape[:2], -1, *ground_truth.shape[-1:])[:, :1, :, :]
            else:
                model_output = ground_truth[:, :1, :, :]
        else:
            model_output = torch.cat((model_output, im), dim=1)
            
    return model_output