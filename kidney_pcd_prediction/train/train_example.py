
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
torch.manual_seed(2)
np.random.seed(2)

# set hyperparameters
n_epochs = 2000
lr = 5e-3
n_points = 100
hidden_dim = 0
num_hidden_layers = 0
depth = 5
train_split = 0.8
invertible = False
test_interval = 25

if torch.cuda.is_available():
    dev = "cuda:0"
elif torch.backends.mps.is_available():
    dev = "mps"
else:
    dev = "cpu"

# set filepath
filepath = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'

# create dataloader
dataset = PointcloudDataset(filepath)
# split dataset into train and test
train_size = int(train_split*len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataloader = DataLoader(train_dataset,batch_size=train_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=test_size,shuffle=True)

# create model
if invertible:
    model = NICEModel(n_points,depth,hidden_dim=hidden_dim,num_hidden_layers=num_hidden_layers).to(dev)
else:
    model = MLP(n_points,depth).to(dev)
print(model.layers)

# create optimizer
optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1e-5)

# create loss function
loss_fn = nn.L1Loss().to(dev)

# create directory to save models
if not os.path.exists('models'):
    os.mkdir('models')

losses = []
test_losses,test_diffs,test_lb,test_difftoavg = [],[],[],[]
train_diffs = []
# train model
for epoch in range(n_epochs):
    model.train()
    for i,(x,lb) in enumerate(dataloader):
        optimizer.zero_grad()
        x = x.to(dev)
        lb = lb.to(dev)
        out = model(x)
        loss = loss_fn(out,lb)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        train_diffs.append(torch.sum(torch.abs(out-lb)).item()/(n_points*3*len(lb)))
        print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch,i,loss.item()),end='')

    #calculate validation loss
    if epoch % test_interval == 0 and epoch != 0:
        model.eval()
        for i, (xt, lbt) in enumerate(test_dataloader):
            xt = xt.to(dev)
            lbt = lbt.to(dev)
            outt = model(xt)
            loss = loss_fn(outt, lbt)
            test_losses.append(loss.item())
            test_diffs.append(torch.sum(torch.abs(outt-lbt)).item()/(n_points*3*len(lbt)))
            test_lb.append(lbt.cpu().detach())
            test_difftoavg.append(torch.sum(torch.abs(lbt)).item()/(n_points*3*len(lbt)))
        print('\nAverage test loss: {}'.format(np.mean(test_losses)))
        # print average test difference for each point, as fraction of pointcloud size
        print('Average distance from label to test pred for each point: {}'.format(test_diffs[-1]))
        print('Average distance from label to average pointcloud for each point: {}'.format(test_difftoavg[-1]))

        print()

import matplotlib.pyplot as plt

fig2 = plt.figure(figsize=(10, 10))
plt.plot(np.arange(1,len(test_diffs)+1)*test_interval, test_diffs,label='Test')
plt.plot(np.arange(1,len(test_difftoavg)+1)*test_interval, test_difftoavg,label='Just Using Average')
plt.plot(np.arange(len(train_diffs)), train_diffs,label='Train')
plt.title('Distance Curve')
plt.xlabel('Epoch')
plt.ylabel('Average distance from label to prediction for each point')
plt.ylim(0,0.1)
plt.legend()
plt.show()

#plot moving average of loss curve, and an input / output pointcloud pair
# extract left and right average pointclouds from the dataset
for i in range(len(xt)):
    left_pointcloud = xt[i].cpu().detach().numpy().reshape(-1,3) + dataset.left_avg.reshape(-1,3)
    if invertible:
        left_pointcloud_pred = model.inverse(lbt)[i].cpu().detach().numpy().reshape(-1,3)+dataset.left_avg.reshape(-1,3)
    right_pointcloud_pred = outt[i].cpu().detach().numpy().reshape(-1,3)+dataset.right_avg.reshape(-1,3)
    right_pointcloud = lbt[i].cpu().detach().numpy().reshape(-1,3)+dataset.right_avg.reshape(-1,3)

    lim = np.abs(left_pointcloud).max()*1.1
    # subplot of 3d pointclouds and loss curves
    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 12))
    ax[0].scatter(left_pointcloud[:, 0], left_pointcloud[:, 1], left_pointcloud[:, 2],label='Actual')
    if invertible:
        ax[0].scatter(left_pointcloud_pred[:, 0], left_pointcloud_pred[:, 1], left_pointcloud_pred[:, 2],label='Predicted')
    ax[0].scatter(dataset.left_avg.reshape(-1,3)[:, 0], dataset.left_avg.reshape(-1,3)[:, 1], dataset.left_avg.reshape(-1,3)[:, 2], label='Average',color='red')
    ax[0].set_title('Left Pointcloud')
    ax[0].set_xlim(-lim,lim)
    ax[0].set_ylim(-lim,lim)
    ax[0].set_zlim(-lim,lim)

    ax[1].scatter(right_pointcloud[:, 0], right_pointcloud[:, 1], right_pointcloud[:, 2],label='Actual')
    ax[1].scatter(right_pointcloud_pred[:, 0], right_pointcloud_pred[:, 1], right_pointcloud_pred[:, 2],label='Predicted')
    ax[1].scatter(dataset.right_avg.reshape(-1,3)[:, 0], dataset.right_avg.reshape(-1,3)[:, 1], dataset.right_avg.reshape(-1,3)[:, 2], label='Average',color='red')
    ax[1].set_title('Right Pointcloud')
    ax[1].set_xlim(-lim,lim)
    ax[1].set_ylim(-lim,lim)
    ax[1].set_zlim(-lim,lim)
    plt.show(block=True)

