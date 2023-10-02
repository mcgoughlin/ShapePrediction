
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
n_epochs = 10000
lr = 5e-5
n_points = 500
depth = 10
dropout = 0.95

# set filepath
filepath = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'

# create dataloader
dataset = PointcloudDataset(filepath)
dataloader = DataLoader(dataset,batch_size=len(dataset),shuffle=False)

# create model
model = MLP(n_points,depth,dropout=dropout).cuda()
print(model.layers)

# create optimizer
optimizer = optim.Adam(model.parameters(),lr=lr)

# create loss function
loss_fn = nn.MSELoss()

# create directory to save models
if not os.path.exists('models'):
    os.mkdir('models')

# train model
for epoch in range(n_epochs):
    for i,(x,lb) in enumerate(dataloader):
        optimizer.zero_grad()
        x = x.cuda()
        lb = lb.cuda()
        out = model(x)
        loss = loss_fn(out,lb)
        loss.backward()
        optimizer.step()
        print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch,i,loss.item()),end='')

# test invertibility post-training for sanity check
# model.eval()
# print(model)
# random = torch.rand((1, 1500)).cuda()
# out = model(random)
# inv_in = model.inverse(out)
# diff = torch.sum(torch.abs(inv_in-random)).item()
# print(diff)
