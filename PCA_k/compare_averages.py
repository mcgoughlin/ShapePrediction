import pandas as pd
from PCA_k.procrustes_utils import find_average,procrustes_analysis
import matplotlib.pyplot as plt
import numpy as np


def chamfer_distance(pc1, pc2):
    dist1 = np.min(np.linalg.norm(pc1[:, np.newaxis] - pc2, axis=2),
                   axis=1)  # calculates the distance from each point in pc1 to the nearest point in pc2
    dist2 = np.min(np.linalg.norm(pc2[:, np.newaxis] - pc1, axis=2),
                   axis=1)  # calculates the distance from each point in pc2 to the nearest point in pc1
    return (np.mean(dist1) + np.mean(dist2)) / 2.0  # returns the average of the two distances for each point cloud


if __name__ == "__main__":
    # obj_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/cleaned_objs'
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    number_of_points = 3000
    n_iter = 20000
    tolerance = 1e-7
    n_components = 10

    df = pd.read_csv(features_csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]
    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]
    df_left = df.loc[df['position'] == 'left']
    df_right = df.loc[df['position'] == 'right']

    data = []
    average_pointclouds = []

    for df, side in zip([df_left,df_right],['Left','Right']):
        entry = {'position':side}
        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        average_pointclouds.append(average_pointcloud)

    #flip the right kidney average pointcloud
    average_pointclouds[1][:,2] = average_pointclouds[1][:,2]*-1

    #rigid alignment of the right kidney average pointcloud to the left kidney average pointcloud
    _, aligned_lr,transforms = procrustes_analysis(average_pointclouds[0], average_pointclouds, include_target=False,return_transformations=True)
    print(transforms)
    #calculate maximum diameter of the left and right average pointclouds
    max_diameter = np.max([np.linalg.norm(aligned_lr[0][i] - aligned_lr[0][j]) for i in range(len(aligned_lr[0])) for j in range(len(aligned_lr[0]))])
    av_diff = chamfer_distance(aligned_lr[0],aligned_lr[1])
    print(max_diameter,av_diff)
    # print the ratio of the average chamfer distance to the maximum diameter
    print(max_diameter/av_diff)

    #plot the aligned_lr pointclouds on same plot in 3D projection
    fig = plt.figure(figsize=(10,10))
    lim = np.abs(aligned_lr[0]).max()*1.1
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(aligned_lr[0][:,0],aligned_lr[0][:,1],aligned_lr[0][:,2],c='r',marker='o')
    ax.scatter(aligned_lr[1][:,0],aligned_lr[1][:,1],aligned_lr[1][:,2],c='b',marker='o')
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)
    ax.set_zlim(-lim,lim)
    plt.show()