
import numpy as np
import SimpleITK as sitk
from scipy.spatial import distance
from skimage.measure import regionprops
import scipy.ndimage as spim
from scipy import stats


def nifti_2_correctarr(im_n):
    aff = im_n.affine
    im = sitk.GetImageFromArray(im_n.get_fdata())
    im.SetOrigin(-aff[:3, 3])
    im.SetSpacing(im_n.header['pixdim'][1:4].tolist())

    ##flips image along correct axis according to image properties
    flip_im = sitk.Flip(im, np.diag(aff[:3, :3] < -0).tolist())
    nda = np.rot90(sitk.GetArrayViewFromImage(flip_im))
    return nda.copy()


def get_masses(binary_arr, vol_thresh, intensity_image=None):
    return [[mass, mass.centroid] for mass in regionprops(spim.label(binary_arr)[0], intensity_image=intensity_image) if
            mass.area > vol_thresh]


def assign_labels_2_kidneys(k, masses):
    if len(masses) == 1:
        closeness_array = np.zeros((len(k)))
        for index in range(len(k)):
            closeness_array[index] = distance.euclidean(k[index], masses[0])

        masses_to_kidney_association = np.array([np.argmin(closeness_array)])
    else:
        closeness_array = np.zeros((len(k), len(masses)))

        for index in range(len(k)):
            for jdex in range(len(masses)):
                closeness_array[index, jdex] = distance.euclidean(k[index], masses[jdex])

        # find each mass's associated kidney, where index in association list corresponds to the tumour's index and
        # the element value corresponds to the kidney's index
        masses_to_kidney_association = np.argmin(closeness_array, axis=0)

    return masses_to_kidney_association


def is_sole_kidney_central(kidney_centroids, im, inf, inplane_spac,
                           test1_length=25, test2_length=10,
                           axes=None):
    sole_kidney = kidney_centroids[0]
    axial, lr_index, ud = axes
    z_bone, lr_bone, ud_bone = np.array(regionprops((im > 250).astype(int))[0].centroid).reshape(-1, 1)[np.array(axes)]
    z_bone, lr_bone, ud_bone = z_bone[0], lr_bone[0], ud_bone[0]

    # assert(1==2)
    # test 1 - does the centre of the single kidney line up with spine within 25mm? if so - central kidney
    if abs(sole_kidney[lr_index] - lr_bone) * inplane_spac < test1_length:
        return True, ud_bone, lr_bone
    else:
        # test 2 - kidney is also central if wraps around spine.
        # does some portion of the kidney wrap around the spine?

        # test 2 distance is 10mm
        _test_extent_inpixels = int((test2_length / 2) * inplane_spac)

        # create test label - where the pixels within +-10mm of centre of bone attenuating tissue are zeroed out
        _central_test = inf
        _central_test[:,
        int(lr_bone - _test_extent_inpixels):int(lr_bone + _test_extent_inpixels),
        :] = 0

        # wrapping is true if one or more objects from test label appear either side of the centre of bone attenuating tissue.
        # if wrapping is true, then the kidney is central.
        _test_centroids = [centroid[lr_index] > lr_bone for _, centroid in get_masses(_central_test, 0)]
        if (False in _test_centroids) and (True in _test_centroids):
            return True, ud_bone, lr_bone
        else:
            return False, ud_bone, lr_bone


def find_orientation(spacing, kidney_centroids, is_axes=True, im=None):
    rounded_spacing = np.around(spacing, decimals=1)

    if not (2 in np.unique(rounded_spacing, return_counts=True)[1]): return 0, 0, 0
    inplane_spacing = stats.mode(rounded_spacing, keepdims=False)[0]
    indices = np.array([0, 1, 2])
    axial = indices[rounded_spacing != inplane_spacing][0]
    if is_axes:
        if axial == 0:
            first_half = im[:, :256]
            second_half = im[:, 256:]
        else:
            first_half = im[:256]
            second_half = im[256:]

        # there should only be one (rough) plane of symmetry in a CT scan: 
        # from the axial perspective that splits the spine in half - leftright when facing person. 
        # Thus, the symmetrical 
        # plane for bones should be up-down. We know the axial plane index already, so to determine 
        # up-down, we simplysplit image along the first non-axial dimension and in half,
        # and compare bone 
        # totals in each half. if these are roughly similar (within 30% of each other) - we 
        # say this is symmetry, and therefore the up-down plane.
        try:
            first_total = np.array(regionprops((first_half > 250).astype(int))[0].area)
        except(IndexError):
            first_total = 0  # index error occurs when not a single bit of bone occurs
        try:
            second_total = np.array(regionprops((second_half > 250).astype(int))[0].area)
        except(IndexError):
            second_total = 0
        fraction = first_total / (second_total + 1e-6)
        if (fraction > 0.7) and (fraction < 1.3):
            if axial == 0:
                lr, ud, = 1, 2
            elif axial == 1:
                lr, ud = 0, 2
            else:
                lr, ud = 0, 1
        else:
            if axial == 0:
                lr, ud, = 2, 1
            elif axial == 1:
                lr, ud = 2, 0
            else:
                lr, ud = 1, 0

        return axial, lr, ud
    # do below when assigning orientations for spacings - where lr and ud distinction doesnt matter
    else:
        return axial, *indices[rounded_spacing == inplane_spacing][::-1]