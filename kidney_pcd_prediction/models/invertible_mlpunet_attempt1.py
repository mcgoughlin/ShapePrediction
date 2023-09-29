import torch
import torch.nn as nn
import copy

# Writing the object class MLPUnet for pointcloud-to-pointcloud mapping. It takes as input a pointcloud and outputs a pointcloud.
# Instances of this class will be initialised with values for pointcloud size, depth, dropout, and activation function.
# This model will be able to switch between left and right kidney mapping, thus will be symmetrical.
# Downsampling and upsampling will be performed with simple linear pytorch layers that double/half the number of points.
# The forward method will take as input a pointcloud and output a pointcloud.
# we want this unet to be able to map from left to right and right to left, so we will have a boolean variable that will switch between the two
# if we want to invert the mapping, the non-linear must be reversible, so we will use a leaky relu activation function

# linear activation function
class linear_activation(nn.Module):
    def __init__(self):
        super(linear_activation,self).__init__()

    def forward(self,x):
        return x

class MLPUnet(nn.Module):
    def __init__(self,n_points=1024,depth=5,dropout=0.9,left_is_input=True):
        super(MLPUnet,self).__init__()
        self.n_points = n_points
        self.depth = depth
        self.dropout = dropout
        self.activation = linear_activation()
        self.layers = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.left_is_input = left_is_input
        # define input and output layers of same dimensionality to input pointcloud at start and end of network
        self.input_layer = nn.Linear(self.n_points,self.n_points)
        self.output_layer = nn.Linear(self.n_points,self.n_points)

        #scheme for downsampling and upsampling is symmetric - we will use the same layers for left and right, except invert for right
        # each layer will have equal number of points in and out
        for i in range(self.depth):
            self.layers.append(nn.Linear(self.n_points//(2**i),self.n_points//(2**(i))))
            nn.init.kaiming_uniform_(self.layers[-1].weight,nonlinearity='relu')

        for i in range(self.depth):
            self.downsample.append(nn.Linear(self.n_points//(2**i),self.n_points//(2**(i+1))))

        for i in range(self.depth-1,-1,-1):
            self.upsample.append(nn.Linear(self.n_points//(2**i),self.n_points//(2**(i))))
            nn.init.kaiming_uniform_(self.upsample[-1].weight,nonlinearity='relu')

        self.dropout = nn.Dropout(self.dropout)

        #create function to extract weights and bias from downsample and upsample, invert them, and reassign to upsample and downsample
    def invert_layer(self,layer):
        # check inversion possible by checking weight matrix determinant is finite and non-zero
        print(torch.det(layer.weight).item())
        assert((torch.det(layer.weight).item() != 0) and (torch.det(layer.weight).abs().item()) != float('inf'))

        invert_weight = torch.inverse(copy.deepcopy(layer.weight))
        # check inversion correct with mat mul
        assert torch.allclose(invert_weight@layer.weight,torch.eye(layer.weight.shape[0],device='cuda'))
        inverse_bias = invert_weight@(-1*copy.deepcopy(layer.bias))
        layer.weight = nn.Parameter(invert_weight)
        layer.bias = nn.Parameter(inverse_bias)
        return layer

    def invert_layers(self, layers,upsample):
        layers_temp = copy.deepcopy(layers)
        for i in range(len(layers)):
            layers_temp[i] = self.invert_layer(upsample[-1*(1+i)])
        for i in range(len(upsample)):
            upsample[i] = self.invert_layer(layers[-1*(1+i)])

        return layers_temp,upsample

    def switch_to_left(self):
        self.left_is_input = True
        self.input_layer,self.output_layer = self.output_layer,self.input_layer
        self.layers, self.upsample = self.invert_layers(self.layers,self.upsample)

    def switch_to_right(self):
        self.left_is_input = False
        self.layers, self.upsample = self.invert_layers(self.layers,self.upsample)


    def forward(self,x):
        x = self.input_layer(x)
        # Downsampling and store each downsampled pointcloud in a list
        downsampled = []
        for i in range(self.depth):
            x = self.layers[i](x)
            x = self.activation(x)
            x = self.downsample[i](x)
            x = self.dropout(x)
            print(x.mean().item())
            downsampled.append(x)

        print('bottom')
        # Upsampling and concatenation
        for i in range(self.depth):
            x = torch.cat((x, downsampled.pop(-1)), dim=1)
            x = self.upsample[i](x)
            x = self.activation(x)
            x = self.dropout(x)
            print(x.mean().item())

        return self.output_layer(x)

# test model if name is main by comparing differences between a forward pass and an inverted forward pass
if __name__ == '__main__':
    model = MLPUnet().cuda()
    random = torch.ones(1,1024).cuda()
    print(model)
    print(1)
    out = model(random)
    print(torch.sum(torch.abs(out)))
    print(out)
    print(out.shape)
    print(2)
    model.switch_to_right()
    inv_in = model(out)
    print(inv_in)
    print(3)
    #compute difference between inverse input and original input
    print(torch.sum(torch.abs(inv_in-random)))
