import torch
import torch.nn as nn

# Writing the object class MLPUnet for pointcloud-to-pointcloud mapping. It takes as input a pointcloud and outputs a pointcloud.
# Instances of this class will be initialised with values for pointcloud size, depth, dropout, and activation function.
# Downsampling and upsampling will be performed with simple linear pytorch layers that double/half the number of points.
# The forward method will take as input a pointcloud and output a pointcloud.
# we want this unet to be able to map from left to right and right to left, so we will have a boolean variable that will switch between the two
# if we want to invert the mapping, the non-linear must be reversible, so we will use a leaky relu activation function
class MLPUnet(nn.Module):
    def __init__(self,n_points=1024,depth=5,dropout=0.9):
        super(MLPUnet,self).__init__()
        self.n_points = n_points
        self.depth = depth
        self.dropout = dropout
        self.activation = nn.LeakyReLU()
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        for i in range(self.depth):
            self.downsample.append(nn.Linear(self.n_points//(2**i),self.n_points//(2**(i+1))))

        self.bottom = nn.Linear(self.n_points//(2**self.depth),self.n_points//(2**self.depth))
        for i in range(self.depth-1,0,-1):
            self.upsample.append(nn.Linear(self.n_points//(2**i),self.n_points//(2**(i))))
        self.dropout = nn.Dropout(self.dropout)
        self.pen = nn.Linear(self.n_points,self.n_points)
        self.out = nn.Linear(self.n_points, self.n_points)

    def forward(self,x):
        # Downsampling and store each downsampled pointcloud in a list
        downsampled = []

        for i in range(self.depth):
            downsampled.append(x)
            x = self.downsample[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        downsampled.append(x)
        # Bottom layer
        x = self.bottom(x)
        # Upsampling and concatenation
        for i in range(self.depth-1):
            x = torch.cat((x, downsampled.pop(-1)), dim=1)
            x = self.upsample[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        x = torch.cat((x, downsampled.pop(-1)), dim=1)
        # Output layer
        x = self.dropout(self.activation(self.pen(x)))
        return self.out(x)


if __name__ == '__main__':
    model = MLPUnet().cuda()
    print(model)
    print(model(torch.randn(1,1024).cuda()).shape)