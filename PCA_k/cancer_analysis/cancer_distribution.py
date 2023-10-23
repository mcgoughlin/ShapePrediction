# for each case's segmentation label in CASE_DATA_PATH and object file in CASE_OBJECT_PATH,
# generate pointcloud from object file and coregister with average kidney pointcloud stored in KIDNEY_AVERAGE_PATH
# then, compare the two pointclouds and calculate the residual between them
# then, if a cancer is present, find the closest point in the average pointcloud to the cancer centroid

import os
import numpy as np
import nibabel as nib
from skimage.measure import regionprops
from scipy.spatial import distance
from scipy import ndimage as ndi
from PCA_k.procrustes_utils import find_average, procrustes_analysis
import scipy.ndimage as spim
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import zoom
import pandas as pd
from PCA_k.cancer_analysis import seglabel_utils as slu
from scipy.optimize import minimize

dataset = 'kits23sncct_objdata'
ov_data = 'kits23'
percentile = 99
number_of_points = 300
n_iter=20000
tolerance=1e-7
dist_to_label = 10
lambda_reg = 1e-3


CASE_IMAGE_PATH = '/media/mcgoug01/nvme/Data/{}/images/'.format(ov_data)
CASE_LABEL_PATH = '/media/mcgoug01/nvme/Data/{}/labels/'.format(ov_data)
CASE_INFNII_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_nii/'.format(dataset)
CASE_INFNPY_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_npy'.format(dataset)
CASE_OBJECT_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/cleaned_objs/'.format(dataset)
CASE_PC_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/aligned_pointclouds_all/'.format(dataset)
FEATURES_CSV_FP = '/media/mcgoug01/nvme/ThirdYear/{}/features_labelled.csv'.format(dataset)

KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_pointcloud.npy'
EIGENPAIR_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/eigenpairs/'



eigenpairs = [np.load(os.path.join(EIGENPAIR_PATH,fp),allow_pickle=True) for fp in os.listdir(EIGENPAIR_PATH)]
dev = norm.ppf(percentile / 100)
average_pointcloud = np.load(KIDNEY_AVERAGE_PATH)
left_count = np.zeros_like(average_pointcloud[:,:2])
right_count = np.zeros_like(average_pointcloud[:,:2])

df = pd.read_csv(FEATURES_CSV_FP)
# find the largest cancer and cyst in each position ['left','right'] in each case and create a separate column to store this information
# cancer and cyst columns are denoted by 'cancer_i_vol' and 'cyst_i_vol' where i is an index betwenn 0 and 9
# if there are less than 10 cancers or cysts, the remaining columns are filled with 0
#do not use 'slu' functions here
# cyst_cols = ['cyst_{}_vol'.format(i) for i in range(10)]
# cancer_cols = ['cancer_{}_vol'.format(i) for i in range(10)]
# df['max_cyst'] = df[cyst_cols].max(axis=1)
# df['max_cancer'] = df[cancer_cols].max(axis=1)

df['max_cyst'] = [0]*len(df)
df['max_cancer'] = [0]*len(df)
df = df[['case','position','max_cyst','max_cancer']]

#check if csv already exists
exists_flag = os.path.exists('/media/mcgoug01/nvme/ThirdYear/{}/residuals.csv'.format(dataset))

for case in df.case.unique():
    print('Processing case {}'.format(case))

    inf_npy = np.load(os.path.join(CASE_INFNPY_PATH, case[:-7]+'.npy'))
    # extract each kidney as a separate label
    kidneys = regionprops(spim.label(inf_npy)[0])
    if len(kidneys) != 2: continue

    #extract voxel data
    im_nib = nib.load(os.path.join(CASE_IMAGE_PATH, case))
    lb_nib = nib.load(os.path.join(CASE_LABEL_PATH, case))
    inf_nib = nib.load(os.path.join(CASE_INFNII_PATH, case))
    # extract spacing
    spacing = lb_nib.header['pixdim'][1:4].tolist()
    lb_data = slu.nifti_2_correctarr(lb_nib)
    im_data = slu.nifti_2_correctarr(im_nib)
    inf_data = slu.nifti_2_correctarr(inf_nib)

    print(lb_data.shape, im_data.shape,inf_npy.shape)
    #resize inf_data to match inf_npy using interpolation function
    lb_reshaped = zoom(lb_data, zoom=(inf_npy.shape[0]/inf_data.shape[0],inf_npy.shape[1]/inf_data.shape[1],inf_npy.shape[2]/inf_data.shape[2]),
                    output=np.uint16, mode='nearest')

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

        largest_cancer = df.loc[df['case'] == case].loc[df['position'] == side]['max_cancer'].values[0]
        largest_cyst = df.loc[df['case'] == case].loc[df['position'] == side]['max_cyst'].values[0]
        if side=='left': og_avg= average_pointcloud
        else: og_avg = np.array([average_pointcloud[:,0], average_pointcloud[:,1],average_pointcloud[:,2]*-1]).T
        # load pc
        pc_data = np.load(os.path.join(CASE_PC_PATH,side, case[:-7]+'.npy'.format(side)))
        #ensure pc_data and avg are aligned, and points have same anatomical meaning by index
        _, [pc_data,avg] = procrustes_analysis(pc_data, [pc_data,og_avg], include_target=False)
        # centre both pointclouds on the origin
        pc_data = pc_data - pc_data.mean(axis=0)
        avg = avg - avg.mean(axis=0)

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

        ALIGNED_OG_AVG = np.copy(og_avg)
        #apply magnitudes of eigenpairs to average pointcloud
        for i in range(3):
            scaled_eigenvector = eigenpairs[i][0] * eigenpairs[i][1] * magnitudes[i]
            ALIGNED_OG_AVG += scaled_eigenvector


        ALIGNED_OG_AVG += centroid
        pc_data = pc_data + centroid

        # if a cancer in the lb is within 10mm of a point in pc_data, add 1 to the cancer count in that index
        # first, dilate the cancer label by 10mm
        cancer_label,cyst_label = np.zeros_like(lb_reshaped),np.zeros_like(lb_reshaped)
        cancer_label[lb_reshaped == 2] = 1
        cyst_label[lb_reshaped == 3] = 1
        cancer_label = ndi.binary_dilation(cancer_label, iterations=dist_to_label//4).astype(np.uint8)
        cyst_label = ndi.binary_dilation(cyst_label, iterations=dist_to_label // 4).astype(np.uint8)
        # then, sample the cancer label with pc_data without using slu functions
        sample_pc_data = np.round(ALIGNED_OG_AVG).astype(int)
        cancer_sample = np.array([cancer_label[x,y,z] if ((x < cyst_label.shape[0]) and (y < cyst_label.shape[1]) and (z < cyst_label.shape[2])) else 0 for x,y,z in sample_pc_data])
        cyst_sample = np.array([cyst_label[x,y,z] if ((x < cyst_label.shape[0]) and (y < cyst_label.shape[1]) and (z < cyst_label.shape[2])) else 0 for x,y,z in sample_pc_data])

        if side == 'left':
            left_count[:, 0] += np.nan_to_num((cancer_sample+1e-9)/(sum(cancer_sample)+1e-9)) # normalise cancer count by size of cancer
            left_count[:, 1] += np.nan_to_num((cyst_sample+1e-9)/(sum(cyst_sample)+1e-9))
        else:
            right_count[:, 0] += np.nan_to_num((cancer_sample+1e-9)/(sum(cancer_sample)+1e-9))
            right_count[:, 1] += np.nan_to_num((cyst_sample+1e-9)/(sum(cyst_sample)+1e-9))

# plot the average pointcloud and use the cancer count to colour the points
# left kidney
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(average_pointcloud[:,0],average_pointcloud[:,1],average_pointcloud[:,2],c=left_count[:,0])
ax.set_title('Left Kidney')
ax = fig.add_subplot(122, projection='3d')
ax.scatter(average_pointcloud[:,0],average_pointcloud[:,1],average_pointcloud[:,2],c=right_count[:,0])
ax.set_title('Right Kidney')
plt.show()
