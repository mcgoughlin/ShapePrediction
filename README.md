## PCA for finding average kidney shape and normal modes of inter-patient kidney shape variation

This was done relatively simply - see nonlinPCA for a script that performs this task. Sigmoid kernel performs best.

## MLP for kidney intra-patient shape prediction. This requires paired kidney pointclouds (L and R) for a patients.

This requires consistency of the meaning of nodes - i.e., node 1 needs to correspond to the same anatomical point in kidney 1, 5 and 100. However, due to the global attentive properties of linear layers, node 1 on left kidney does *not* need to correspond to the same anatomical location on kidney 2. Thus, node-meaning between kidneys need only be consistent between the populations of left and right kidney.

It would be nice of the MLPUnet was invertible/reversible. I.e, it could be trained on left kidney-to-right kidney prediction, and then be used to predict left kidneys from right kidneys, or visa versa, or a mixture of both simultaneously. I find this idea intuitive, as the relationship should be inherently invertible. However, to do this, we need an invertible MLPUnet architecture. This requires symmetry, with no pooling layers.

It seems as though the variation in the left kidney geometry is not predictive of the variation in right kidney geometry (variation wrt. their average). Next, it will be useful to distinguish whether there is any significant shape difference between right and left kidneys.

## Assessing Shape Difference between Kidneys

We can assess the correlation between the right and left kidney by extacting the average left/right kidney pointclouds at *very* high point density (think 3000 points), and comparing the mean chamfer distances between these two averages. At extremely high point density, the differences between these pointclouds should attenuate if they are similar shapes. If not, there should be some significant distance that the average difference between both pointclouds asymptotically approaches as pointcloud density increase. Crucial point here: what classifies as a 'significant distance'? We address this problem in 'compare_averages.py'.

We find that, at 3000 points per pointcloud, the average distance between points in the average left and right kidney is 66.7 times smaller than the maximum diameter of the left kidney. There are real differences in shape here - the left and right kidney (as extracted out of kits23 data) are not perfectly similar - the left kidney appears to be 'beanier' - have a larger dent at the entrance of the main vein, than the right kidney. also, the right kidney appears to be slightly smaller on average.
