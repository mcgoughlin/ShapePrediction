import pandas as pd
from PCA_k.procrustes_utils import find_average_normalised
import numpy as np
import os

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
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/features_unlabelled.csv'

    pointcloud_save_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/aligned_pointclouds'
    number_of_points = 500
    n_iter = 10000
    tolerance = 1e-7

    df = pd.read_csv(features_csv_fp)

    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']
    data = []

    for df, side in zip([df_left,df_right],['left','right']):
        entry = {'position':side}
        cases = df['case'].values
        average_pointcloud,aligned_pointclouds = find_average_normalised(df, obj_folder,number_of_points, n_iter, tolerance)

        # save aligned pointclouds
        save_aligned_pointclouds(aligned_pointclouds,cases,side,pointcloud_save_folder)

        # save average pointcloud
        np.save(pointcloud_save_folder + '/average_{}.npy'.format(side),average_pointcloud)
