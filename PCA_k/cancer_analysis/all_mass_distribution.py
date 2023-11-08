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
number_of_points = 1000
n_iter=20000
tolerance=1e-7
dist_to_label = 4 #mm
lambda_reg = 1e-3
testing = True
cancer_radius_filter = 20 #mm

CASE_IMAGE_PATH = '/media/mcgoug01/nvme/Data/{}/images/'.format(ov_data)
CASE_LABEL_PATH = '/media/mcgoug01/nvme/Data/{}/labels/'.format(ov_data)
CASE_INFNII_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_nii/'.format(dataset)
CASE_INFNPY_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_npy'.format(dataset)
CASE_OBJECT_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/cleaned_objs/'.format(dataset)
CASE_PC_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/aligned_pointclouds_all/'.format(dataset)
FEATURES_CSV_FP = '/media/mcgoug01/nvme/ThirdYear/{}/features_labelled.csv'.format(dataset)

KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_pointcloud.npy'
EIGENPAIR_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/eigenpairs/'
SAVE_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/'.format(dataset)

eigenpairs = [np.load(os.path.join(EIGENPAIR_PATH,fp),allow_pickle=True) for fp in os.listdir(EIGENPAIR_PATH)]
dev = norm.ppf(percentile / 100)
average_pointcloud = np.load(KIDNEY_AVERAGE_PATH)
average_pointcloud = average_pointcloud - average_pointcloud.mean(axis=0)
left_count = np.zeros_like(average_pointcloud[:,:2])
right_count = np.zeros_like(average_pointcloud[:,:2])
lim = average_pointcloud.max()*1.1
canc_thresh = (4/3)*np.pi*(cancer_radius_filter**3)

df = pd.read_csv(FEATURES_CSV_FP)
assert False
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
        # except error if side left/right does not exist
        try:
            largest_cancer = df.loc[df['case'] == case].loc[df['position'] == side]['max_cancer'].values[0]
            largest_cyst = df.loc[df['case'] == case].loc[df['position'] == side]['max_cyst'].values[0]
        except:
            continue
        if side=='left': side_average = average_pointcloud.copy()
        else: side_average = np.array([average_pointcloud.copy()[:,0], average_pointcloud.copy()[:,1],average_pointcloud.copy()[:,2]*-1]).T
        # load pc
        pc_data = np.load(os.path.join(CASE_PC_PATH,side, case[:-7]+'.npy'.format(side)))
        pc_data = pc_data - pc_data.mean(axis=0)
        #ensure pc_data and avg are aligned, and points have same anatomical meaning by index
        _, [pc_data, side_average],mapping = procrustes_analysis(pc_data, [pc_data, side_average], include_target=False,
                                                                 return_mapping=True)
        # centre both pointclouds on the origin
        pc_data = pc_data - pc_data.mean(axis=0)
        side_average = side_average - side_average.mean(axis=0)

        magnitudes = [0,0,0]
        for i in range(3):
            def loss(magnitude):
                average = np.copy(side_average)

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

        eigen_kidney = np.copy(side_average)
        #apply magnitudes of eigenpairs to average pointcloud
        for i in range(3):
            scaled_eigenvector = eigenpairs[i][0] * eigenpairs[i][1] * magnitudes[i]
            eigen_kidney += scaled_eigenvector


        eigen_kidney += centroid
        pc_data = pc_data + centroid

        # if a cancer in the lb is within 10mm of a point in pc_data, add 1 to the cancer count in that index
        # first, dilate the cancer label by 10mm
        cancer_label,cyst_label = np.zeros_like(lb_reshaped),np.zeros_like(lb_reshaped)
        cancer_label[lb_reshaped == 2] = 1
        cyst_label[lb_reshaped == 3] = 1
        cancer_label = ndi.binary_dilation(cancer_label, iterations=dist_to_label//4).astype(np.uint8)
        cyst_label = ndi.binary_dilation(cyst_label, iterations=dist_to_label // 4).astype(np.uint8)
        # then, sample the cancer label with pc_data without using slu functions
        sample_pc_data = np.round(pc_data).astype(int)
        cancer_sample = np.array([cancer_label[x,y,z] if ((x < cyst_label.shape[0]) and (y < cyst_label.shape[1]) and (z < cyst_label.shape[2])) else 0 for x,y,z in sample_pc_data])
        cyst_sample = np.array([cyst_label[x,y,z] if ((x < cyst_label.shape[0]) and (y < cyst_label.shape[1]) and (z < cyst_label.shape[2])) else 0 for x,y,z in sample_pc_data])

        reverse_mapping = np.zeros_like(mapping)
        for i in range(len(mapping)):
            reverse_mapping[mapping[i]] = i

        cancer_sample = cancer_sample[reverse_mapping]
        cyst_sample = cyst_sample[reverse_mapping]

        # smooth counts between neighbouring nodes according to proximity in the average kidney
        cancer_sample = np.array([np.mean(cancer_sample[np.where(distance.cdist([average_pointcloud[i]], average_pointcloud) < dist_to_label//2)[1]]) for i in range(len(average_pointcloud))])
        cyst_sample = np.array([np.mean(cyst_sample[np.where(distance.cdist([average_pointcloud[i]], average_pointcloud) < dist_to_label//2)[1]]) for i in range(len(average_pointcloud))])

        if side == 'left':
            left_count[:, 0] += cancer_sample
            left_count[:, 1] += cyst_sample
        else:
            right_count[:, 0] += cancer_sample
            right_count[:, 1] += cyst_sample

        # plot the average kidney and the target kidney, highlighting 3 random points in the same colour in both pointclouds
        #each point is a different colour, but the same colour in both pointclouds
        if testing:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(121, projection='3d')
            ax.scatter(average_pointcloud[:, 0], average_pointcloud[:, 1], average_pointcloud[:, 2], c=left_count[:, 0])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_title('Left Kidney')
            ax = fig.add_subplot(122, projection='3d')
            ax.scatter(average_pointcloud[:, 0], average_pointcloud[:, 1], average_pointcloud[:, 2],
                       c=right_count[:, 0])
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.set_zlim(-lim, lim)
            ax.set_title('Right Kidney')
            plt.show(block=True)

#save the cancer and cyst counts as pointcloud npy files for left, right, and both added together
both_count = left_count + right_count
np.save(SAVE_PATH + 'left_count.npy',left_count)
np.save(SAVE_PATH + 'right_count.npy',right_count)
np.save(SAVE_PATH + 'both_count.npy',both_count)


# # plot the average pointcloud and use the cancer count to colour the points
# # left kidney
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(average_pointcloud[:,0],average_pointcloud[:,1],average_pointcloud[:,2],c=left_count[:,0])
#set axes limits
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_title('Left Kidney')
ax = fig.add_subplot(122, projection='3d')
ax.scatter(average_pointcloud[:,0],average_pointcloud[:,1],average_pointcloud[:,2],c=right_count[:,0])
#set axes limits
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_title('Right Kidney')
plt.show()
