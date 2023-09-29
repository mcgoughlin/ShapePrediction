
import matplotlib.pyplot as plt
import pandas as pd
from PCA_k.procrustes_utils import find_average
import numpy as np

if __name__ == "__main__":
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    number_of_points = 1024
    n_iter = 20000
    tolerance = 1e-7

    df = pd.read_csv(features_csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]

    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']

    for df in [df_left,df_right]:
        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(average_pointcloud[:,0], average_pointcloud[:,1], average_pointcloud[:,2])
        lim = np.abs(average_pointcloud).max()
        ax.set_xlim(-lim,lim)
        ax.set_ylim(-lim,lim)
        ax.set_zlim(-lim,lim)

        plt.show(block=True)