import torch
# import torch dataloader
import os
from torch.utils.data import Dataset
import numpy as np

# this script is a torch dataset for pointclouds, that facilitates MLPUnet training
# it takes as input a filepath for folders containing npy files of left kidney pointclouds and right kidney pointclouds
# it outputs a torch dataset that can be used to train the MLPUnet model
# the dataloader will output a left kidney pointcloud input and a batch of right kidney pointcloud as label
# the dataloader will also be able to output a right kidney pointcloud input and a batch of left kidney pointcloud as label

class PointcloudDataset(Dataset):
    def __init__(self,filepath):
        self.filepath = filepath
        # save object variable for left and right filepath
        self.left_filepath = self.filepath + '/left/'
        self.right_filepath = self.filepath + '/right/'
        # load filenames of left and right pointclouds
        self.left_filenames = os.listdir(self.left_filepath)
        self.right_filenames = os.listdir(self.right_filepath)

        # get rid of pointclouds that don't exists in both left and right folders
        self.left_filenames = [filename for filename in self.left_filenames if filename in self.right_filenames]
        self.right_filenames = [filename for filename in self.right_filenames if filename in self.left_filenames]

        # load average left and right pointclouds
        self.left_avg = np.load(os.path.join(filepath,'average_left.npy')).reshape(1,-1)
        self.right_avg = np.load(os.path.join(filepath,'average_right.npy')).reshape(1,-1)
        self.n_samples = len(self.right_filepath)
        self.left_is_input = True

    def __len__(self):
        return len(self.left_filenames)

    def switch_to_left(self):
        self.left_is_input = True

    def switch_to_right(self):
        self.left_is_input = False

    def __getitem__(self,idx):
        # load left and right pointclouds - make sure shape is (1,number of points)
        left_pointcloud = np.load(self.left_filepath + self.left_filenames[idx]).reshape(1,-1)
        right_pointcloud = np.load(self.right_filepath + self.right_filenames[idx]).reshape(1,-1)

        #subtract average pointcloud from each pointcloud and convert to torch tensors
        left_pointcloud = torch.from_numpy(left_pointcloud - self.left_avg).float()[0]
        right_pointcloud = torch.from_numpy(right_pointcloud - self.right_avg).float()[0]

        # augmentation - add noise in x,y,z directions in proportion to the standard deviation of the pointcloud
        std_var_aug = 0.0005
        left_pointcloud += torch.randn_like(left_pointcloud)*std_var_aug
        right_pointcloud += torch.randn_like(right_pointcloud)*std_var_aug

        # if left is input, return left as input and right as label
        if self.left_is_input:
            return left_pointcloud, right_pointcloud
        else:
            return right_pointcloud, left_pointcloud

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'
    dataset = PointcloudDataset(path)
    print(len(dataset))
    # display the average left and right point clouds into two seperate 3d scatter plots
    left_avg = dataset.left_avg.reshape(-1,3)
    right_avg = dataset.right_avg.reshape(-1,3)
    lim = np.max(np.abs(left_avg))

    fig, ax = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 5))
    ax[0].scatter(left_avg[:,0],left_avg[:,1],left_avg[:,2])
    ax[1].scatter(right_avg[:,0],right_avg[:,1],right_avg[:,2])

    #set lims
    for i in range(2):
        ax[i].set_xlim(-lim,lim)
        ax[i].set_ylim(-lim,lim)
        ax[i].set_zlim(-lim,lim)
        ax[i].view_init(-20, 45)

    #link viewing angle
    def on_move(event):
        if event.inaxes == ax[1]:
            ax[0].view_init(elev=ax[1].elev, azim=ax[1].azim)
        else:
            ax[1].view_init(elev=ax[0].elev, azim=ax[0].azim)
        fig.canvas.draw_idle()

    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
    plt.show()

    a,b = dataset[0]
