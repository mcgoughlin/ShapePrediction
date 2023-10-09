from kidney_pcd_prediction.dataloader.pointcloud_dataloader import PointcloudDataset
import matplotlib.pyplot as plt
import numpy as np

#this script will assess the correlation between the left and right kidney pointclouds
#it will do this by applying a manova test between the two sets of pointclouds
# set filepath
filepath = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'
# create dataloader
dataset = PointcloudDataset(filepath)

#want to assess the correlation between right and left kidney pointclouds
# split dataset into right and left pointclouds

left_pointclouds = []
right_pointclouds = []
for i in range(len(dataset)):
    print(dataset[i][0].shape)
    left_pointclouds.append(dataset[i][0])
    right_pointclouds.append(dataset[i][1])

left_pointclouds = np.array(left_pointclouds)
right_pointclouds = np.array(right_pointclouds)

#now we have the left and right pointclouds, we can apply a manova test
#we will use the manova test from the statsmodels library
from statsmodels.multivariate.manova import MANOVA
