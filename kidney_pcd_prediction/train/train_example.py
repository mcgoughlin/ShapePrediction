
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
n_epochs = 300
lr = 5e-4
n_points = 500
hidden_dim = n_points
num_hidden_layers = 0
batch_size = 32
depth = 4
dropout = 0.95
train_split = 0.8

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
dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_dataloader = DataLoader(test_dataset,batch_size=1,shuffle=True)

# create model
model = NICEModel(n_points,depth,dropout=dropout,hidden_dim=hidden_dim,num_hidden_layers=num_hidden_layers).to(dev)
print(model.layers)

# create optimizer
optimizer = optim.Adam(model.parameters(),lr=lr)

# create loss function
loss_fn = nn.MSELoss().to(dev)

# create directory to save models
if not os.path.exists('models'):
    os.mkdir('models')

# test invertibility pre-training for sanity check
model.eval()
print('Testing invertibility pre-training')
random = torch.rand((1, 1500)).to(dev)
out = model(random)
inv_in = model.inverse(out)
diff = torch.sum(torch.abs(inv_in-random)).item()
print(diff)


losses = []
# train model
for epoch in range(n_epochs):
    for i,(x,lb) in enumerate(dataloader):
        optimizer.zero_grad()
        x = x.to(dev)
        lb = lb.to(dev)
        out = model(x)
        loss = loss_fn(out,lb)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        print('\rEpoch: {}, Batch: {}, Loss: {}'.format(epoch,i,loss.item()),end='')

# find average loss and distance between input and output pointclouds
model.eval()
test_losses, test_diffs = [],[]
for i,(x,lb) in enumerate(test_dataloader):
    x = x.to(dev)
    lb = lb.to(dev)
    out = model(x)
    loss = loss_fn(out, lb)
    test_losses.append(loss.item())
    test_diffs.append(torch.sum(torch.abs(out-lb)).item())

print('\nAverage test loss: {}'.format(np.mean(test_losses)))
print('Average test diff for whole pointcloud: {}'.format(np.mean(test_diffs)))
# print average test difference for each point, as fraction of pointcloud size
print('Average test diff for each point: {}'.format(np.mean(test_diffs)/n_points))



# test invertibility post-training for sanity check
model.eval()
print('post-training invertibility test')
random = torch.rand((1, 1500)).to(dev)
out = model(random)
inv_in = model.inverse(out)
diff = torch.sum(torch.abs(inv_in-random)).item()
print(diff)

#plot moving average of loss curve, and an input / output pointcloud pair
# extract left and right average pointclouds from the dataset
import matplotlib.pyplot as plt

left_pointcloud = x[0].cpu().detach().numpy().reshape(-1,3) + dataset.left_avg.reshape(-1,3)
left_pointcloud_pred = model.inverse(lb)[0].cpu().detach().numpy().reshape(-1,3) + dataset.left_avg.reshape(-1,3)
right_pointcloud_pred = out[0].cpu().detach().numpy().reshape(-1,3) + dataset.right_avg.reshape(-1,3)
right_pointcloud = lb[0].cpu().detach().numpy().reshape(-1,3) + dataset.right_avg.reshape(-1,3)

lim = np.abs(left_pointcloud).max()*1.1
# subplot of 3d pointclouds and loss curves
fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 12))
ax[0].scatter(left_pointcloud[:, 0], left_pointcloud[:, 1], left_pointcloud[:, 2],label='Actual')
ax[0].scatter(left_pointcloud_pred[:, 0], left_pointcloud_pred[:, 1], left_pointcloud_pred[:, 2],label='Predicted')
ax[0].set_title('Left Pointcloud')
ax[0].set_xlim(-lim,lim)
ax[0].set_ylim(-lim,lim)
ax[0].set_zlim(-lim,lim)
ax[1].scatter(right_pointcloud[:, 0], right_pointcloud[:, 1], right_pointcloud[:, 2],label='Actual')
ax[1].scatter(right_pointcloud_pred[:, 0], right_pointcloud_pred[:, 1], right_pointcloud_pred[:, 2],label='Predicted')
ax[1].set_title('Right Pointcloud')
ax[1].set_xlim(-lim,lim)
ax[1].set_ylim(-lim,lim)
ax[1].set_zlim(-lim,lim)
ax[2].plot(zs=np.arange(len(losses)),xs=losses,ys=losses)
ax[2].set_title('Loss Curve')
plt.show()