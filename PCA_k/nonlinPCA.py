import pandas as pd
from PCA_k.procrustes_utils import find_average
import numpy as np
from sklearn.decomposition import KernelPCA as PCA
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm

if __name__ == "__main__":
    # obj_folder = '/media/mcgoug01/nvme/ThirdYear/CTORG_objdata/cleaned_objs'
    obj_folder = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/cleaned_objs'
    features_csv_fp = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/features_labelled.csv'
    number_of_points = 1000
    n_iter = 20000
    tolerance = 1e-7
    n_components = 10
    visualise_component=3
    percentile_principal_comp_variation = 99
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

    data = []

    for df, side in zip([df_left,df_right],['Left','Right']):
        entry = {'position':side}
        average_pointcloud,aligned_pointclouds = find_average(df, obj_folder,number_of_points, n_iter, tolerance)
        aligned_shape = aligned_pointclouds.shape
        variance_point_clouds = np.array([pc - average_pointcloud for pc in aligned_pointclouds])
        variance_point_clouds = variance_point_clouds.reshape(variance_point_clouds.shape[0],-1).T
        cov_matrix = np.cov(variance_point_clouds)
        pca.fit(cov_matrix)
        entry['components'] = pca.eigenvectors_.reshape((n_components,aligned_shape[1],aligned_shape[2]))
        entry['std_dev'] = np.sqrt(pca.eigenvalues_)
        entry['average_pointcloud'] = average_pointcloud.reshape((aligned_shape[1],aligned_shape[2]))

        print(np.around(pca.eigenvalues_/np.sum(pca.eigenvalues_), decimals=3))
        print(np.around(np.cumsum(pca.eigenvalues_/np.sum(pca.eigenvalues_)), decimals=3))
        print(np.around(entry['std_dev'], decimals=3))
        data.append(entry)

        # plot the average point cloud and +/- 1 standard deviation of the first 2 components, where the central plot is just the average point cloud
        # in a 3x3 grid
        lim = np.abs(average_pointcloud).max()*1.1
        fig, ax = plt.subplots(1, visualise_component, subplot_kw={'projection': '3d'}, figsize=(20, 12))
        #reduce gaps between subplots
        plt.subplots_adjust(wspace=0, hspace=0)
        fig.suptitle('Principal Components of {} Kidney'.format(side))
        graphs = []
        animations = []
        eigen_pairs = []
        for component_index in range(0, visualise_component):
            index = np.argsort(pca.eigenvalues_)[-(component_index+1)]
            eigenvalue = pca.eigenvalues_[index]
            eigenvector = pca.eigenvectors_[:,index].reshape((aligned_shape[1],aligned_shape[2]))
            eigen_pairs.append((eigenvalue,eigenvector))
            # create figure with 3 subplots in one row with a 3d projection
            title = ax[component_index].set_title('PC {} of {}'.format(component_index + 1, visualise_component))
            lim = np.abs(average_pointcloud).max()*1.1
            graphs.append(ax[component_index].scatter(average_pointcloud[:,0], average_pointcloud[:,1], average_pointcloud[:,2]))
            #set the axes properties
            ax[component_index].set_xlim3d(-lim, lim)
            ax[component_index].set_ylim3d(-lim, lim)
            ax[component_index].set_zlim3d(-lim, lim)
            sin_curve = np.sin(2*np.pi*np.linspace(0.0000001, 1, animation_frames))

        def update_graph(num):
            alternating_mag = sin_curve[num]
            for i in range(visualise_component):
                eigenvalue,eigenvector = eigen_pairs[i]
                component_of_variation = eigenvector * np.sqrt(eigenvalue) * alternating_mag * norm.ppf(percentile_principal_comp_variation / 100)  # multiplies eigenvector by the std dev. corresponding to previously defined percentile
                graphs[i]._offsets3d = (average_pointcloud[:,0] + component_of_variation[:, 0],
                                 average_pointcloud[:,1] + component_of_variation[:,1],
                                 average_pointcloud[:,2] + component_of_variation[:,2])
            return graphs


        def on_move(event):
            for i in range(visualise_component):
                ax[i].elev = event.inaxes.elev
                ax[i].azim = event.inaxes.azim
            fig.canvas.draw_idle()

        animations.append(animation.FuncAnimation(fig, update_graph, animation_frames,interval=40, blit=False))
        c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

        plt.show(block=True)