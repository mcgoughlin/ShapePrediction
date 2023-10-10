import pandas as pd
from PCA_k.procrustes_utils import find_average
import numpy as np
from sklearn.decomposition import KernelPCA as PCA
import matplotlib.pyplot as plt
from scipy.stats import norm

if __name__ == "__main__":
    # obj_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/cleaned_objs'
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    number_of_points = 1000
    n_iter = 20000
    tolerance = 1e-7
    n_components = 10
    visualise_compontents=4
    percentile_principal_comp_variation = 95

    df = pd.read_csv(features_csv_fp)

    # Select rows where all cyst and cancer measurements are 0
    columns_to_check = ['cyst_{}_vol'.format(i) for i in range(10)] + ['cancer_{}_vol'.format(i) for i in range(10)]
    df = df[df[columns_to_check].sum(axis=1) == 0]
    pca = PCA(kernel = 'rbf', n_components=n_components)
    # extract case and kidney position data, split between left and right
    df = df[['case', 'position']]

    data = []

    average_pointcloud,aligned_pointclouds = find_average(df, obj_folder, number_of_points, n_iter, tolerance)
    aligned_shape = aligned_pointclouds.shape
    variance_point_clouds = np.array([pc - average_pointcloud for pc in aligned_pointclouds])
    variance_point_clouds = variance_point_clouds.reshape(variance_point_clouds.shape[0],-1).T
    cov_matrix = np.cov(variance_point_clouds)
    pca.fit(cov_matrix)

    print(pca.eigenvalues_)
    print(np.around(pca.eigenvalues_/np.sum(pca.eigenvalues_), decimals=3))
    print(np.around(np.cumsum(pca.eigenvalues_/np.sum(pca.eigenvalues_)), decimals=3))
    print(np.around(np.sqrt(pca.eigenvalues_), decimals=3))

    # plot the average point cloud and +/- 1 standard deviation of the first 2 components, where the central plot is just the average point cloud
    # in a 3x3 grid
    lim = np.abs(average_pointcloud).max()*1.1

    for component_index in range(1,visualise_compontents+1):
        fig, ax = plt.subplots(1, 3, subplot_kw={'projection': '3d'}, figsize=(20, 12))
        plt.subplots_adjust(wspace=0, hspace=0)
        index = np.argsort(pca.eigenvalues_)[-component_index]
        eigenvalue = pca.eigenvalues_[index]
        eigenvector = pca.eigenvectors_[:,index].reshape((aligned_shape[1],aligned_shape[2]))
        component_of_variation = eigenvector * np.sqrt(eigenvalue) * norm.ppf(percentile_principal_comp_variation / 100) # 95th percentile
        colour_one = np.linalg.norm(component_of_variation,axis=1)
        colour_two = np.linalg.norm(component_of_variation,axis=1)*-1

        ax[0].scatter(average_pointcloud[:,0]+component_of_variation[:,0], average_pointcloud[:,1]+component_of_variation[:,1],
                          average_pointcloud[:,2]+component_of_variation[:,2],c=colour_one,vmin=colour_two.min(),vmax=colour_one.max())

        ax[1].scatter(average_pointcloud[:, 0], average_pointcloud[:, 1],
                           average_pointcloud[:, 2])

        ax[2].scatter(average_pointcloud[:,0]-component_of_variation[:,0], average_pointcloud[:,1]-component_of_variation[:,1],
                          average_pointcloud[:,2]-component_of_variation[:,2],c=colour_two,vmin=colour_two.min(),vmax=colour_one.max())
        for j in range(3):
            ax[j].set_xlim(-lim, lim)
            ax[j].set_ylim(-lim, lim)
            ax[j].set_zlim(-lim, lim)
            ax[j].view_init(-20, 45)

        def on_move(event):
            if event.inaxes == ax[1]:
                ax[0].view_init(elev=ax[1].elev, azim=ax[1].azim)
                ax[2].view_init(elev=ax[1].elev, azim=ax[1].azim)
            elif event.inaxes == ax[0]:
                ax[1].view_init(elev=ax[0].elev, azim=ax[0].azim)
                ax[2].view_init(elev=ax[0].elev, azim=ax[0].azim)
            else:
                ax[1].view_init(elev=ax[2].elev, azim=ax[2].azim)
                ax[0].view_init(elev=ax[2].elev, azim=ax[2].azim)
            fig.canvas.draw_idle()

        fig.suptitle('component '+str(component_index),fontsize=20)
        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)
        plt.show(block=True)