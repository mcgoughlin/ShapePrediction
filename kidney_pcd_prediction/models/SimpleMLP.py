# write a python class that implements a simple MLP model
# the model should take as input a pointcloud and output a pointcloud

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_points, depth, dropout=0.95):
        super(MLP, self).__init__()
        self.n_points = n_points
        self.depth = depth
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_points*3,n_points*3))
        for i in range(depth-1):
            self.layers.append(nn.Linear(n_points*3,n_points*3))
        self.final = nn.Linear(n_points*3,n_points*3)
        self.skips = nn.ModuleList([nn.Linear(n_points*3,n_points*3) for i in range(depth-1)])

    def forward(self, x):
        for i in range(self.depth-1):
            x = self.layers[i](x)
            x = F.relu(x)
            x = F.dropout(x + self.skips[i](x),self.dropout)
        x = self.final(x)
        return x