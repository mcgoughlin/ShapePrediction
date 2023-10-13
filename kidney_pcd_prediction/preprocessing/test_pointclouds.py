#we want to plot the average pointcloud and then an example two pointclouds,
#we want to highlight the same index points in each pointcloud, to make sure their alignment is correct

import numpy as np
import os

path = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'
average_left_pointcloud = np.load(path + '/average_left.npy')
average_right_pointcloud = np.load(path + '/average_right.npy')

left_pointcloud1 = np.load(path + '/left/10.npy')
left_pointcloud2 = np.load(path + '/left/11.npy')
left_pointcloud3 = np.load(path + '/left/15.npy')

# plot left point clouds and mark 3 random points with different colours that match between the two point clouds

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#plot 3 pointclouds with the average pointcloud
fig = plt.figure()
#three subplots
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132, projection='3d')
ax3 = fig.add_subplot(133, projection='3d')

#remove space between subplots
fig.subplots_adjust(wspace=0,hspace=0)

#plot the average pointcloud
ax1.scatter(average_left_pointcloud[:,0],average_left_pointcloud[:,1],average_left_pointcloud[:,2],label='Average')
#plot the first pointcloud
ax1.scatter(left_pointcloud1[:,0],left_pointcloud1[:,1],left_pointcloud1[:,2],label='Pointcloud 1')

#plot the average pointcloud
ax2.scatter(average_left_pointcloud[:,0],average_left_pointcloud[:,1],average_left_pointcloud[:,2],label='Average')
#plot the second pointcloud
ax2.scatter(left_pointcloud2[:,0],left_pointcloud2[:,1],left_pointcloud2[:,2],label='Pointcloud 2')

#plot the average pointcloud
ax3.scatter(average_left_pointcloud[:,0],average_left_pointcloud[:,1],average_left_pointcloud[:,2],label='Average')
#plot the third pointcloud
ax3.scatter(left_pointcloud3[:,0],left_pointcloud3[:,1],left_pointcloud3[:,2],label='Pointcloud 3')

# loop through the plots and highlight 3 random points in each plot
random_point_indices = np.random.randint(0,len(left_pointcloud2),3)
#generate three different colours for each point
colors = ['red','blue','green']
for ax,pc in zip([ax1,ax2,ax3],[left_pointcloud1,left_pointcloud2,left_pointcloud3]):
    ax.scatter(pc[random_point_indices,0],pc[random_point_indices,1],pc[random_point_indices,2],label='Random Points',color=colors,s=100)
    #plot these points as black dots on the average pointcloud
    ax.scatter(average_left_pointcloud[random_point_indices,0],average_left_pointcloud[random_point_indices,1],average_left_pointcloud[random_point_indices,2],label='Random Points',color='black',s=100)

plt.show()
