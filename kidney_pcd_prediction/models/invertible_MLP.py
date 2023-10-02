import torch
import torch.nn as nn
import torch.nn.init as init
from kidney_pcd_prediction.models.invertible_mlp_layers import AdditiveCouplingLayer

# adapted from https://github.com/paultsw/nice_pytorch/blob/master/nice/models.py
# from paper: https://arxiv.org/pdf/1410.8516.pdf

def _build_relu_network(latent_dim, hidden_dim, num_hidden_layers):
    """Helper function to construct a ReLU network of varying number of layers."""
    if num_hidden_layers == 0 or hidden_dim == 0:
        return nn.Sequential(nn.Linear(latent_dim, latent_dim),nn.ReLU(), nn.BatchNorm1d(latent_dim))

    _modules = [nn.Linear(latent_dim, hidden_dim)]
    for _ in range(num_hidden_layers):
        _modules.append(nn.Linear(hidden_dim, hidden_dim))
        _modules.append(nn.ReLU())
        _modules.append(nn.BatchNorm1d(hidden_dim))
    _modules.append(nn.Linear(hidden_dim, latent_dim))
    return nn.Sequential(*_modules)


class NICEModel(nn.Module):
    """
    Replication of model from the paper:
      "Nonlinear Independent Components Estimation",
      Laurent Dinh, David Krueger, Yoshua Bengio (2014)
      https://arxiv.org/abs/1410.8516

    Contains the following components:
    * four additive coupling layers with nonlinearity functions consisting of
      five-layer RELUs
    * a diagonal scaling matrix output layer
    """

    def __init__(self, num_points, num_layers, hidden_dim=0,
                 num_hidden_layers=0, dropout=0.9):
        super(NICEModel, self).__init__()
        assert (num_points % 2 == 0), "[NICEModel] only even input dimensions supported for now"
        assert(num_layers >= 3), "[NICEModel] num_layers must be above 2"
        self.input_dim = num_points*3
        half_dim = int(self.input_dim / 2)
        self.dropout = nn.Dropout(dropout)
        self.num_hidden = num_hidden_layers
        self.layers = []

        is_odd = True
        for i in range(num_layers):
            if is_odd:
                self.layers.append(AdditiveCouplingLayer(self.input_dim, 'odd', _build_relu_network(half_dim, hidden_dim, self.num_hidden)))
            else:
                self.layers.append(AdditiveCouplingLayer(self.input_dim, 'even', _build_relu_network(half_dim, hidden_dim, self.num_hidden)))
            is_odd = not is_odd

        self.layers = nn.Sequential(*self.layers)
        self.scaling_diag = nn.Parameter(torch.ones(self.input_dim))

        # randomly initialize weights:
        for layer in self.layers:
            for p in layer.parameters():
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p, nonlinearity='relu')
                else:
                    init.normal_(p, mean=0., std=0.001)

    def forward(self, xs):
        """
        Forward pass through all invertible coupling layers.

        Args:
        * xs: float tensor of shape (B,dim).

        Returns:
        * ys: float tensor of shape (B,dim).
        """
        for i in range(len(self.layers)):
            xs = self.layers[i](xs)
            xs = self.dropout(xs)
        ys = torch.matmul(xs, torch.diag(torch.exp(self.scaling_diag)))
        return ys

    def inverse(self, ys):
        """Invert a set of draws from gaussians"""
        with torch.no_grad():
            xs = torch.matmul(ys, torch.diag(torch.reciprocal(torch.exp(self.scaling_diag))))
            for i in range(len(self.layers)):
                xs = self.layers[-1*(1+i)].inverse(xs)
        return xs

# test invertiblility of this model if run as main
if __name__ == '__main__':
    model = NICEModel(1024,5).cuda()
    model.eval()
    print(model)
    random = torch.rand((1, 1024)).cuda()
    out = model(random)
    inv_in = model.inverse(out)
    diff = torch.sum(torch.abs(inv_in-random)).item()
    print(diff)
    print(model.inverse(out)-random)