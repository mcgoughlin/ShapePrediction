# for each case's segmentation label in CASE_DATA_PATH and object file in CASE_OBJECT_PATH,
# generate pointcloud from object file and coregister with average kidney pointcloud stored in KIDNEY_AVERAGE_PATH
# then, compare the two pointclouds and calculate the residual between them
# then, if a cancer is present, find the closest point in the average pointcloud to the cancer centroid

import os
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
from scipy.spatial import distance
from PCA_k.procrustes_utils import find_average, procrustes_analysis
import scipy.ndimage as spim
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import zoom
import pandas as pd
from PCA_k.cancer_analysis import seglabel_utils as slu
from scipy.optimize import minimize

CASE_IMAGE_PATH = '/media/mcgoug01/nvme/Data/kits23/images/'
CASE_LABEL_PATH = '/media/mcgoug01/nvme/Data/kits23/labels/'
CASE_INFNII_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/inferences_nii/'
CASE_INFNPY_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/inferences_npy'
KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_pointcloud.npy'
EIGENPAIR_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/eigenpairs/'
CASE_OBJECT_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs/'
CASE_PC_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds_all/'
features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
percentile = 99

number_of_points = 300
n_iter=20000
tolerance=1e-7
dist_to_label = 2.5
lambda_reg = 1e-3

eigenpairs = [np.load(os.path.join(EIGENPAIR_PATH,fp),allow_pickle=True) for fp in os.listdir(EIGENPAIR_PATH)]
dev = norm.ppf(percentile / 100)
average_pointcloud = np.load(KIDNEY_AVERAGE_PATH)
cancer_count = np.zeros_like(average_pointcloud)

df = pd.read_csv(features_csv_fp)
df = df[['case']]


for case in df.case.unique():
    print('Processing case {}'.format(case))
    #extract voxel data
    im_nib = nib.load(os.path.join(CASE_IMAGE_PATH, case))
    lb_nib = nib.load(os.path.join(CASE_LABEL_PATH, case))
    inf_nib = nib.load(os.path.join(CASE_INFNII_PATH, case))
    inf_npy = np.load(os.path.join(CASE_INFNPY_PATH, case[:-7]+'.npy'))
    # extract spacing
    spacing = lb_nib.header['pixdim'][1:4].tolist()
    lb_data = slu.nifti_2_correctarr(lb_nib)
    im_data = slu.nifti_2_correctarr(im_nib)
    inf_data = slu.nifti_2_correctarr(inf_nib)

    #resize inf_data to match inf_npy using interpolation function
    lb_reshaped = zoom(lb_data, zoom=(inf_npy.shape[0]/inf_data.shape[0],inf_npy.shape[1]/inf_data.shape[1],inf_npy.shape[2]/inf_data.shape[2]),
                    output=np.uint16, mode='nearest')

    print(lb_data.shape, im_data.shape, lb_reshaped.shape,inf_npy.shape)
    # extract each kidney as a separate label
    kidneys = regionprops(spim.label(inf_npy)[0])
    centroids = np.array([kidney.centroid for kidney in kidneys])
    bboxs = np.array([kidney.bbox for kidney in kidneys])
    spacing_axes = slu.find_orientation(spacing, centroids, is_axes = False)

    if spacing_axes == (0, 0, 0): continue
    z_spac, inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]
    axes = slu.find_orientation(inf_data.shape, centroids, is_axes=True, im=im_data)
    if axes == (0, 0, 0): continue
    axial, lr, ud = axes
    # use lr axis to extract left and right kidney centroid and bbox
    left_kidney = 1 if centroids[0][lr] < centroids[1][lr] else 0
    right_kidney = 1 if centroids[0][lr] >= centroids[1][lr] else 0

    left_kidney_centroid = centroids[left_kidney]
    right_kidney_centroid = centroids[right_kidney]
    left_kidney_bbox = bboxs[left_kidney]
    right_kidney_bbox = bboxs[right_kidney]

    # loop through each kidney side, centroid, and bbox
    for side, centroid, bbox in zip(['left', 'right'], [left_kidney_centroid, right_kidney_centroid], [left_kidney_bbox, right_kidney_bbox]):
        if side=='left': og_avg= average_pointcloud
        else: og_avg = np.array([average_pointcloud[:,0], average_pointcloud[:,1],average_pointcloud[:,2]*-1]).T
        # load pc
        pc_data = np.load(os.path.join(CASE_PC_PATH,side, case[:-7]+'.npy'.format(side)))
        #ensure pc_data and avg are aligned, and points have same anatomical meaning by index
        _, [pc_data,avg] = procrustes_analysis(pc_data, [pc_data,og_avg], include_target=False)
        magnitudes = [0,0,0]
        for i in range(3):
            def loss(magnitude):
                average = np.copy(avg)

                for j in range(i): # add all previously scaled eigenvectors
                    scaled_eigenvector = eigenpairs[j][0] * eigenpairs[j][1] * magnitudes[j]
                    average += scaled_eigenvector

                scaled_eigenvector = eigenpairs[i][0] * eigenpairs[i][1] * magnitude
                average += scaled_eigenvector

                distances = np.min(distance.cdist(average, pc_data, 'euclidean'),axis=0)
                mean_distance = np.mean(distances)
                return mean_distance + lambda_reg*(magnitude**2)

            res = minimize(loss, [0], method='Nelder-Mead', options={'maxiter':50000,'disp':False},tol=1e-6)
            magnitudes[i] = res.x[0]

        #apply magnitudes of eigenpairs to average pointcloud
        for i in range(3):
            scaled_eigenvector = eigenpairs[i][0] * eigenpairs[i][1] * magnitudes[i]
            avg += scaled_eigenvector


        avg += centroid
        pc_data = pc_data + centroid

        #calculate residual
        distances = np.min(distance.cdist(avg, pc_data, 'euclidean'),axis=0)
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)
        print('Mean distance: {}'.format(mean_distance))
        print('Max distance: {}'.format(max_distance))


