import numpy as np
import matplotlib.pyplot as plt

# load average pointcloud
LEFT_KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_left_pointcloud.npy'
RIGHT_KIDNEY_AVERAGE_PATH = '/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/aligned_pointclouds/average_right_pointcloud.npy'
left_average_pointcloud = np.load(LEFT_KIDNEY_AVERAGE_PATH)
right_average_pointcloud = np.load(RIGHT_KIDNEY_AVERAGE_PATH)
#load cancer counts
both_counts = np.load('/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/add_and_kits_both_count.npy')[:,0]
left_counts = np.load('/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/add_and_kits_left_count.npy')[:,0]/1.64
right_counts = np.load('/media/mcgoug01/nvme/ThirdYear/kits23sncct_objdata/add_and_kits_right_count.npy')[:,0]/1.68

counts = np.concatenate([left_counts,right_counts],axis=0)

min_value = min(np.min(left_counts), np.min(right_counts))
max_value = max(np.max(left_counts), np.max(right_counts))


averages = np.concatenate(
    [np.array([left_average_pointcloud[:, 2]+10, left_average_pointcloud[:, 1], left_average_pointcloud[:, 0]]).T,
     np.array([right_average_pointcloud[:, 2]-10, right_average_pointcloud[:, 1], right_average_pointcloud[:, 0] ]).T],
    axis=0)
lim = np.abs(averages).max() * 1.1
counts = np.concatenate([left_counts, right_counts], axis=0)
fig = plt.figure(figsize=(15, 7.5))
ax = fig.add_subplot(111, projection='3d')
min_value = counts.min()
max_value = counts.max()
sc = ax.scatter(averages[:, 0], averages[:, 2], averages[:, 1]*-1, cmap='jet', vmin=min_value, vmax=max_value,
                c=counts,s=50)
# PLOT TEXT LABEL IN 3D TO HIGHLIGHT LEFT AND RIGHT KIDNEY
ax.text(averages.max()+10, 0, 0, 'Left Kidney')
ax.text(averages.max() * -1 - 12, 0, 0, 'Right Kidney')
# plot arrow in centre of image point towards the negative z axis
ax.quiver(0,0,0,0,0,10,length=3,arrow_length_ratio=0.1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_zlim(-lim, lim)
cbar = fig.colorbar(sc, ax=fig.axes, shrink=0.5)
cbar.set_label('Cancer Rate / %')
plt.show(block=True)


