
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from kidney_pcd_prediction.models.invertible_MLP import NICEModel
from kidney_pcd_prediction.models.SimpleMLP import MLP
from kidney_pcd_prediction.dataloader.pointcloud_dataloader import PointcloudDataset
import numpy as np
import os

# this script trains the NICE model on the CTORG dataset
# it takes as input a filepath for folders containing npy files of left kidney pointclouds and right kidney pointclouds
# it outputs a trained NICE model
# the dataloader will output a left kidney pointcloud input and a batch of right kidney pointcloud as label
# the dataloader will also be able to output a right kidney pointcloud input and a batch of left kidney pointcloud as label

# set random seed
torch.manual_seed(0)
np.random.seed(0)

# set hyperparameters
n_epochs = 1000
lr = 5e-4
n_points = 500
hidden_dim = n_points
num_hidden_layers = 1
depth = 5
dropout = 0.8

# set filepath
filepath = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'

# create dataloader
dataset = PointcloudDataset(filepath)
dataloader = DataLoader(dataset,batch_size=len(dataset),shuffle=False)

# create model
model = NICEModel(n_points,depth,dropout=dropout,hidden_dim=hidden_dim,num_hidden_layers=num_hidden_layers).cuda()
print(model.layers)

# create optimizer
optimizer = optim.Adam(model.parameters(),lr=lr)

# create loss function
loss_fn = nn.MSELoss()

# create directory to save models
if not os.path.exists('models'):
    os.mkdir('models')

# test invertibility post-training for sanity check
model.eval()
print(model)
random = torch.rand((1, 1500)).cuda()
out = model(random)
inv_in = model.inverse(out)
diff = torch.sum(torch.abs(inv_in-random)).item()
print(diff)


losses = []
# train model
for epoch in range(n_epochs):
    for i,(x,lb) in enumerate(dataloader):
        optimizer.zero_grad()
        x = x.cuda()
        lb = lb.cuda()
        out = model(x)
        loss = loss_fn(out,lb)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch,i,loss.item()),end='')

#plot moving average of loss curve
import matplotlib.pyplot as plt
sliding_window = len(losses)//100
plt.plot(losses)
plt.plot(np.convolve(losses,np.ones(sliding_window)/sliding_window,mode='valid'))
plt.show()


