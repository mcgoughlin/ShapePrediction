import pandas as pd
from PCA_k.procrustes_utils import find_average, procrustes_analysis
import numpy as np
from sklearn.decomposition import KernelPCA as PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
import os

if __name__ == "__main__":
    # obj_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/cleaned_objs'
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    save_path = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'

    number_of_points = 1000
    n_iter = 20000
    tolerance = 1e-7
    n_components = 10
    visualise_component=3
    percentile_principal_comp_variation = 0.99
    kernel = 'sigmoid'
    animation_frames = 20

    df = pd.read_csv(features_csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]
    pca = PCA(kernel = kernel, n_components=n_components)
    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']
    pcs = []
    data = []
    averages = []
    for df, side in zip([df_left,df_right],['Left','Right']):
        entry = {'position':side}
        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        #extend pcs list with the aligned pointclouds
        pcs.append(aligned_pointclouds)
        #append the average pointcloud to the averages list
        averages.append(average_pointcloud)

    # plot both averages on the same plot in 3D projection
    fig = plt.figure(figsize=(10,10))
    lim = np.abs(averages[0]).max()*2
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(averages[0][:,0],averages[0][:,1],averages[0][:,2],c='r',marker='o')
    ax.scatter(averages[1][:,0],averages[1][:,1],averages[1][:,2]-20,c='b',marker='o')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)
    ax.set_title('Average Pointclouds of Left and Right Kidneys')
    plt.show(block=True)

    fig = plt.figure(figsize=(10,10))
    lim = np.abs(averages[0]).max()*1.1
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(averages[0][:,0],averages[0][:,1],averages[0][:,2],c='r',marker='o')
    ax.scatter(averages[1][:,0],averages[1][:,1],averages[1][:,2],c='b',marker='o')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)
    ax.set_title('Average Pointclouds of Left and Right Kidneys')
    plt.show(block=True)

    #save the average pointclouds
    np.save(os.path.join(save_path,'average_left_pointcloud.npy'),averages[0])
    np.save(os.path.join(save_path,'average_right_pointcloud.npy'),averages[1])


    for pc_group, side, average_pointcloud in zip(pcs,['Left','Right'],averages):
        variance_point_clouds = np.array([pc - average_pointcloud for pc in pc_group])
        cov_matrix = np.cov(variance_point_clouds.reshape(variance_point_clouds.shape[0],-1).T)
        pca.fit(cov_matrix)
        aligned_shape = pc_group.shape
        for component_index in range(0, visualise_component):
            index = np.argsort(pca.eigenvalues_)[-(component_index + 1)]
            eigenvalue = pca.eigenvalues_[index]
            eigenvector = pca.eigenvectors_[:, index].reshape((aligned_shape[1], aligned_shape[2]))
            eigen_pair = (eigenvalue, eigenvector)
            np.save(os.path.join(save_path,'eigenpairs','{}_eigenpair_{}.npy'.format(side,component_index)),np.asarray(eigen_pair,dtype='object'))

    #flip the right kidney pointclouds
    rgt_pcs = np.array([np.array([pc[:,0], pc[:,1],pc[:,2]*-1]).T for pc in pcs[1]])
    #concatenate the left and right pointclouds
    all_pcs = np.concatenate((pcs[0],rgt_pcs),axis=0)
    average_pointcloud, aligned_pointclouds = procrustes_analysis(average_pointcloud, all_pcs, include_target=False)

    # save the average pointcloud
    np.save(os.path.join(save_path,'average_pointcloud.npy'),average_pointcloud)
    aligned_shape = aligned_pointclouds.shape
    variance_point_clouds = np.array([pc - average_pointcloud for pc in aligned_pointclouds])
    variance_point_clouds = variance_point_clouds.reshape(variance_point_clouds.shape[0],-1).T
    cov_matrix = np.cov(variance_point_clouds)
    pca.fit(cov_matrix)

    for component_index in range(0, visualise_component):
        index = np.argsort(pca.eigenvalues_)[-(component_index+1)]
        eigenvalue = pca.eigenvalues_[index]
        eigenvector = pca.eigenvectors_[:,index].reshape((aligned_shape[1],aligned_shape[2]))
        eigen_pair = (eigenvalue, eigenvector)
        np.save(os.path.join(save_path, 'eigenpairs', 'eigenpair_{}.npy'.format( component_index)), np.asarray(eigen_pair,dtype='object'))