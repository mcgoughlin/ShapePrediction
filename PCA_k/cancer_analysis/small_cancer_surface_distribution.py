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
dist_to_label = 8 #mm
lambda_reg = 1e-3
testing = False
cancer_radius_filter = 20 #mm

LEFT_KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_left_pointcloud.npy'
RIGHT_KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_right_pointcloud.npy'
EIGENPAIR_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/eigenpairs/'
left_eigenpairs = [np.load(os.path.join(EIGENPAIR_PATH, fp), allow_pickle=True) for fp in os.listdir(EIGENPAIR_PATH)
                   if fp.startswith('Left')]
right_eigenpairs = [np.load(os.path.join(EIGENPAIR_PATH, fp), allow_pickle=True) for fp in os.listdir(EIGENPAIR_PATH)
                    if fp.startswith('Right')]

dev = norm.ppf(percentile / 100)
left_average_pointcloud = np.load(LEFT_KIDNEY_AVERAGE_PATH)
right_average_pointcloud = np.load(RIGHT_KIDNEY_AVERAGE_PATH)
left_average_pointcloud -= left_average_pointcloud.mean(axis=0)
right_average_pointcloud -= right_average_pointcloud.mean(axis=0)

left_count = np.zeros_like(left_average_pointcloud[:, 0])
right_count = np.zeros_like(right_average_pointcloud[:, 0])
lim = left_average_pointcloud.max() * 1.1
canc_thresh = (4 / 3) * np.pi * (cancer_radius_filter ** 3)

right_cases = 0
left_cases = 0
missed_left = 0
missed_right = 0

for dataset,ov_data in zip(['kits23sncct_objdata','add_objdata'],['kits23','Addenbrookes']):

    CASE_IMAGE_PATH = '/media/mcgoug01/nvme/Data/{}/images/'.format(ov_data)
    CASE_LABEL_PATH = '/media/mcgoug01/nvme/Data/{}/labels/'.format(ov_data)
    CASE_INFNII_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_nii/'.format(dataset)
    CASE_INFNPY_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/inferences_npy'.format(dataset)
    CASE_OBJECT_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/cleaned_objs/'.format(dataset)
    CASE_PC_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/aligned_pointclouds_all/'.format(dataset)
    FEATURES_CSV_FP = '/media/mcgoug01/nvme/ThirdYear/{}/features_labelled.csv'.format(dataset)
    SAVE_PATH = '/media/mcgoug01/nvme/ThirdYear/{}/'.format(dataset)

    df = pd.read_csv(FEATURES_CSV_FP)
    #find largest cancer for each case - if there are less than 10 cancers, fill the remaining columns with 0
    # find the largest cancer and cyst in each position ['left','right'] in each case and create a separate column to store this information
    # cancer and cyst columns are denoted by 'cancer_i_vol' and 'cyst_i_vol' where i is an index betwenn 0 and 9
    # if there are less than 10 cancers or cysts, the remaining columns are filled with 0
    #do not use 'slu' functions here
    cancer_cols = ['cancer_{}_vol'.format(i) for i in range(10)]
    df['max_cancer'] = df[cancer_cols].max(axis=1)

    df = df[['case','position','max_cancer']]
    # count the number of cases with cancers in each position
    # if a case has a cancer in both positions, it is counted twice


    og_df = df.copy()

    #exclude cases with cancers larger than 20mm
    df = df.loc[df['max_cancer'] < canc_thresh]
    # exclude cases where position is not left or right
    df = df.loc[df['position'].isin(['left','right'])]

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
        for side, centroid, bbox, side_average,eigenpairs in zip(['left', 'right'], [left_kidney_centroid, right_kidney_centroid], [left_kidney_bbox, right_kidney_bbox],
                                                           [left_average_pointcloud.copy(), right_average_pointcloud.copy()],[left_eigenpairs,right_eigenpairs]):
            # except error if side left/right does not exist
            try:
                largest_cancer = df.loc[df['case'] == case].loc[df['position'] == side]['max_cancer'].values[0]
                if largest_cancer == 0: continue
                else:
                    if side == 'left': left_cases += 1
                    else: right_cases += 1
            except:
                continue

            # if dataset is addenbrookes, swap y and z axis
            if dataset == 'add_objdata':
                temp = side_average[:, 1].copy()
                side_average[:, 1] = side_average[:, 2].copy()
                side_average[:, 2] = temp

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
            cancer_label = np.zeros_like(lb_reshaped)
            cancer_label[lb_reshaped == 2] = 1
            cancer_label = ndi.binary_dilation(cancer_label, iterations=dist_to_label//4).astype(np.uint8)
            # then, sample the cancer label with pc_data without using slu functions
            sample_pc_data = np.round(pc_data).astype(int)
            cancer_sample = np.array([cancer_label[x,y,z] if ((x < cancer_label.shape[0]) and (y < cancer_label.shape[1]) and (z < cancer_label.shape[2])) else 0 for x,y,z in sample_pc_data])



            reverse_mapping = np.zeros_like(mapping)
            for i in range(len(mapping)):
                reverse_mapping[mapping[i]] = i

            cancer_sample = cancer_sample[reverse_mapping]
            # smooth counts between neighbouring nodes according to proximity in the average kidney
            cancer_sample = np.array([np.mean(cancer_sample[np.where(distance.cdist([side_average[reverse_mapping][i]], side_average[reverse_mapping]) < dist_to_label//4)[1]]) for i in range(len(side_average))])

            if cancer_sample.sum() < 2:
                if side == 'left': missed_left += 1
                else: missed_right += 1
                continue

            if side == 'left':
                left_count += cancer_sample
            else:
                right_count += cancer_sample

            # plot the average kidney and the target kidney, highlighting 3 random points in the same colour in both pointclouds
            #each point is a different colour, but the same colour in both pointclouds
            if testing:
                averages = np.concatenate([np.array([left_average_pointcloud[:, 0], left_average_pointcloud[:, 1], left_average_pointcloud[:, 2]+10]).T,
                                            np.array([right_average_pointcloud[:, 0], right_average_pointcloud[:, 1], right_average_pointcloud[:, 2]-10]).T], axis=0)
                counts = np.concatenate([left_count[:,0], right_count[:,0]], axis=0)
                fig = plt.figure(figsize=(15, 7.5))
                ax = fig.add_subplot(111, projection='3d')
                min_value = counts.min()
                max_value = counts.max()
                sc = ax.scatter(averages[:, 0], averages[:, 1], averages[:, 2], cmap='jet', vmin=min_value, vmax=max_value,
                                c=counts)
                # PLOT TEXT LABEL IN 3D TO HIGHLIGHT LEFT AND RIGHT KIDNEY
                ax.text(0, 0, averages.max() + 2, 'Left Kidney')
                ax.text(0, 0, averages.max() * -1 - 2, 'Right Kidney')
                ax.set_xlim(-lim, lim)
                ax.set_ylim(-lim, lim)
                ax.set_zlim(-lim, lim)
                cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.5)
                cbar.set_label('Cancer Count')
                plt.show(block=True)


                global index
                index = np.argmax(np.sum(lb_reshaped == 2, axis=(0, 1)))
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111)
                ax.imshow(lb_reshaped[:, index], cmap='gray',vmin=0,vmax=2)

                if dataset == 'add_objdata':
                    pc_data_slice = pc_data[np.where(np.round(pc_data[:,2]) == index)]
                    pc_data_colour = cancer_sample[mapping][np.where(np.round(pc_data[:,2]) == index)]
                    eigen_kidney_slice = eigen_kidney[np.where(np.round(eigen_kidney[:,2]) == index)]
                    ax.imshow(lb_reshaped[:, :, index], cmap='gray')
                    ax.scatter(pc_data_slice[:, 1], pc_data_slice[:, 0], c=pc_data_colour, s=5)
                    ax.scatter(eigen_kidney_slice[:, 1], eigen_kidney_slice[:, 0], c='b', s=5)
                else:
                    pc_data_slice = pc_data[np.where(np.round(pc_data[:,1]) == index)]
                    pc_data_colour = cancer_sample[mapping][np.where(np.round(pc_data[:,1]) == index)]
                    eigen_kidney_slice = eigen_kidney[np.where(np.round(eigen_kidney[:,1]) == index)]
                    ax.scatter(pc_data_slice[:, 2], pc_data_slice[:, 0], c=pc_data_colour, s=5)
                    ax.scatter(eigen_kidney_slice[:, 2], eigen_kidney_slice[:, 0], c='b', s=5)

                def scroll(event):
                    global index
                    if event.button == 'up':
                        index = (index + 1) % lb_reshaped.shape[2]
                    else:
                        index = (index - 1) % lb_reshaped.shape[2]
                    ax.clear()
                    pc_data_slice = pc_data[np.where(np.round(pc_data[:, 2]) == index)]
                    pc_data_colour = cancer_sample[mapping][np.where(np.round(pc_data[:, 2]) == index)]
                    eigen_kidney_slice = eigen_kidney[np.where(np.round(eigen_kidney[:, 2]) == index)]
                    ax.imshow(lb_reshaped[:, :, index], cmap='gray',vmin=0,vmax=2)
                    ax.scatter(pc_data_slice[:, 1], pc_data_slice[:, 0], c=pc_data_colour, s=5)
                    ax.scatter(eigen_kidney_slice[:, 1], eigen_kidney_slice[:, 0], c='b', s=5)
                    fig.canvas.draw_idle()

                fig.canvas.mpl_connect('scroll_event', scroll)
                plt.show(block=True)

#save the cancer counts as pointcloud npy files for left, right, and both added together
both_count = left_count + right_count
np.save(SAVE_PATH + 'add_and_kits_left_count.npy',left_count)
np.save(SAVE_PATH + 'add_and_kits_right_count.npy',right_count)
np.save(SAVE_PATH + 'add_and_kits_both_count.npy',both_count)


# # plot the average pointcloud and use the cancer count to colour the points
# # left kidney
print('Left cases: {}'.format(left_cases))
print('Right cases: {}'.format(right_cases))
print('Missed left cases: {}'.format(missed_left))
print('Missed right cases: {}'.format(missed_right))

averages = np.concatenate(
    [np.array([left_average_pointcloud[:, 2]+10, left_average_pointcloud[:, 1], left_average_pointcloud[:, 0]]).T,
     np.array([right_average_pointcloud[:, 2]-10, right_average_pointcloud[:, 1], right_average_pointcloud[:, 0] ]).T],
    axis=0)
counts = np.concatenate([left_count, right_count], axis=0)
fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(111, projection='3d')
min_value = counts.min()
max_value = counts.max()
sc = ax.scatter(averages[:, 0], averages[:, 2], averages[:, 1], cmap='jet', vmin=min_value, vmax=max_value,
                c=counts,s=50)
# PLOT TEXT LABEL IN 3D TO HIGHLIGHT LEFT AND RIGHT KIDNEY
ax.text(averages.max()+2 , 0, 0, 'Left Kidney')
ax.text(averages.max() * -1 - 7, 0, 0, 'Right Kidney')
# plot arrow in centre of image point towards the negative z axis
ax.quiver(0,0,0,0,0,-10,length=3,arrow_length_ratio=0.1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.5)
cbar.set_label('Cancer Count')
plt.show(block=True)
