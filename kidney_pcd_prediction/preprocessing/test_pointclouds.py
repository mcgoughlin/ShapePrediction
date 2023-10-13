#we want to plot the average pointcloud and then an example two pointclouds,
#we want to highlight the same index points in each pointcloud, to make sure their alignment is correct

import numpy as np
import os

path = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'
average_left_pointcloud = np.load(path + '/average_left.npy')
average_right_pointcloud = np.load(path + '/average_right.npy')

left_pointcloud = np.load(path + '/left/10.npy')
right_pointcloud = np.load(path + '/right/10.npy')

# plot left point clouds and mark 3 random points with different colours that match between the two point clouds

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.scatter(right_pointcloud[:,0],right_pointcloud[:,1],right_pointcloud[:,2],label='Right Pointcloud')
ax.scatter(average_right_pointcloud[:,0],average_right_pointcloud[:,1],average_right_pointcloud[:,2],label='Average Right Pointcloud')

# pick 3 random points from the left pointcloud
left_pointcloud_indices = np.random.choice(left_pointcloud.shape[0],3,replace=False)

# plot the 3 random points in the left pointcloud
ax.scatter(right_pointcloud[left_pointcloud_indices,0],right_pointcloud[left_pointcloud_indices,1],right_pointcloud[left_pointcloud_indices,2],label='Left Pointcloud Indices',c='red',s=100)
#plot the 3 random points in the average left pointcloud
ax.scatter(average_right_pointcloud[left_pointcloud_indices,0],average_right_pointcloud[left_pointcloud_indices,1],average_right_pointcloud[left_pointcloud_indices,2],label='Average Left Pointcloud Indices',c='black',s=100)
ax.set_title('Left Pointcloud')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()