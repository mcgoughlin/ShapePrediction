# for each case's segmentation label in CASE_DATA_PATH and object file in CASE_OBJECT_PATH,
# generate pointcloud from object file and coregister with average kidney pointcloud stored in KIDNEY_AVERAGE_PATH
# then, compare the two pointclouds and calculate the residual between them
# then, if a cancer is present, find the closest point in the average pointcloud to the cancer centroid

import os
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
import scipy.ndimage as spim
import open3d as o3d
import PCA_k.cancer_analysis.object_creation_utils as ocu

CASE_DATA_PATH = '/media/mcgoug01/nvme/Data/kits23/labels/'
KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_pointcloud.npy'