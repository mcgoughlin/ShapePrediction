import pandas as pd
from PCA_k.procrustes_utils import find_average, procrustes_analysis
import numpy as np
import os
from sklearn.decomposition import KernelPCA as PCA

#function save aligned pointsclouds as npy files, with a csv file containing the case and kidney position
def save_aligned_pointclouds(aligned_pointclouds,cases,position,home_folder):
    # save aligned pointclouds into a left and right subdirectory
    # in the save_folder, save a csv file containing the case and kidney position

    # create left and right subdirectories
    if position == 'left':
        save_folder = home_folder + '/left'
    else:
        save_folder = home_folder + '/right'

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # save aligned pointclouds
    results = []
    for i in range(len(cases)):
        entry = {'position': position,
                 'case': cases[i].split('.')[0],
                 'filepath': save_folder + '/{}.npy'.format(cases[i].split('.')[0])}
        case = entry['case']
        pointcloud = aligned_pointclouds[i]
        np.save(save_folder + '/{}.npy'.format(entry['case']),pointcloud)
        results.append(entry)

    # save case, position, and filepath to csv. if csv already exists, append to it. if there are duplicates within
    # the existing csv and the new csv, remove the duplicates
    df = pd.DataFrame(results)
    if os.path.exists(home_folder + '/pcd_metadata.csv'):
        df_old = pd.read_csv(home_folder + '/pcd_metadata.csv')
        df = pd.concat([df,df_old])
        df = df.drop_duplicates()
    df.to_csv(home_folder + '/pcd_metadata.csv',index=False)


if __name__ == "__main__":
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'

    pointcloud_save_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds'
    number_of_points = 300
    n_iter = 10000
    tolerance = 1e-7
    n_components = 10
    keep_components = 3
    kernel = 'sigmoid'

    pca = PCA(kernel=kernel, n_components=n_components)
    df = pd.read_csv(features_csv_fp)
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]

    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']
    pcs = []
    data = []

    for df, side in zip([df_left,df_right],['left','right']):
        entry = {'position':side}
        cases = df['case'].values

        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        #extend pcs list with the aligned pointclouds
        pcs.append(aligned_pointclouds)
        save_aligned_pointclouds(aligned_pointclouds,cases,side,pointcloud_save_folder)


    #flip the right kidney pointclouds
    rgt_pcs = np.array([np.array([pc[:,0], pc[:,1],pc[:,2]*-1]).T for pc in pcs[1]])
    #concatenate the left and right pointclouds
    all_pcs = np.concatenate((pcs[0],rgt_pcs),axis=0)
    average_pointcloud, aligned_pointclouds = procrustes_analysis(average_pointcloud, all_pcs, include_target=False)

    # save the average pointcloud and each aligned pointcloud
    np.save(pointcloud_save_folder + '/average_pointcloud.npy', average_pointcloud)
    save_aligned_pointclouds(aligned_pointclouds, cases, side, pointcloud_save_folder)
    aligned_shape = aligned_pointclouds.shape
    variance_point_clouds = np.array([pc - average_pointcloud for pc in aligned_pointclouds])
    variance_point_clouds = variance_point_clouds.reshape(variance_point_clouds.shape[0], -1).T
    cov_matrix = np.cov(variance_point_clouds)
    pca.fit(cov_matrix)

    for component_index in range(keep_components):
        index = np.argsort(pca.eigenvalues_)[-(component_index+1)]
        eigenvalue = pca.eigenvalues_[index]
        eigenvector = pca.eigenvectors_[:,index].reshape((aligned_shape[1],aligned_shape[2]))
        eigenpair = np.array([eigenvalue,eigenvector],dtype=object)

        np.save(pointcloud_save_folder + '/eigenpairs/eigenpair_{}.npy'.format(component_index),eigenpair)



