import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from scipy.spatial import distance
from skimage.measure import regionprops, marching_cubes
import scipy.ndimage as spim
from scipy import stats
import test_utils as tu
import feature_extraction_utils as feu
import graph_smoothing_utils as gmu
import file_utils as fu
import csv


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


def seg_2_mesh(segmentation, axes=None, show=False):
    axial_index, lr, ud = axes
    if show:
        index = segmentation.shape[axial_index] // 2
        if axial_index == 1:
            plt.imshow(segmentation[:, index])
        elif axial_index == 2:
            plt.imshow(segmentation[:, :, index])
        else:
            plt.imshow(segmentation[index])

    verts, faces, norm, val = marching_cubes(segmentation > 0, 0.8, step_size=1, allow_degenerate=True)
    if show:
        show_verts = np.round(verts)
        show_verts = show_verts[show_verts[:, axial_index] == index]
        plt.scatter(show_verts[:, lr], show_verts[:, ud])
        plt.show(block=True)
    return np.array([verts, faces], dtype=object)


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


def create_labelled_dataset(path, dataset, im_p, infnpy_p, infnii_p, lb_p,
                            is_testing=False, size_thresh=200, overwrite=True):
    home = os.path.join(path, 'objects')
    save_dir = os.path.join(home, dataset)
    fu.create_folder(home), fu.create_folder(save_dir)
    rawv_p, rawo_p, cleano_p, c_p, v_p, e_p = fu.setup_save_folders(save_dir)

    cases = [case for case in os.listdir(im_p) if case.endswith('.nii.gz')]
    cases.sort()

    feature_fp = os.path.join(save_dir, 'features_labelled.csv')
    if overwrite:
        access = 'w'
        csv_exists = False
    else:
        access = 'a'
        csv_exists = os.path.exists(feature_fp)
    with open(feature_fp, access, newline="") as feature_file:
        for case_index, case in enumerate(cases):
            ########### LOAD DATA #############
            print(case)
            if csv_exists:
                with open(feature_fp, "r") as csv_check:
                    written_cases = [row['case'] for row in csv.DictReader(csv_check)]
                    if len(written_cases) == 0:
                        csv_exists = False
                    elif case in written_cases:
                        continue

            inf_n = nib.load(os.path.join(infnii_p, case))
            inf = nifti_2_correctarr(inf_n)
            kid_data = np.array(get_masses(inf > 0, 20), dtype=object)
            if len(kid_data) == 0: continue

            im_n = nib.load(os.path.join(im_p, case))
            inf_4mm = np.load(os.path.join(infnpy_p, case[:-7] + '.npy'), allow_pickle=True)
            lb_n = nib.load(os.path.join(lb_p, case))

            im = nifti_2_correctarr(im_n)
            lb = nifti_2_correctarr(lb_n)

            spacing = inf_n.header['pixdim'][1:4]
            spacing_axes = find_orientation(spacing, kid_data[:, 1], is_axes=False)
            if spacing_axes == (0, 0, 0): continue
            z_spac, inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]
            axes = find_orientation(im.shape, kid_data[:, 1], is_axes=True, im=im)
            if axes == (0, 0, 0): continue
            axial, lr, ud = axes
            vox_volmm = np.prod(spacing)

            if axial == 0:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / z_spac, 4 / inplane_spac, 4 / inplane_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])
            elif axial == 1:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / inplane_spac, 4 / z_spac, 4 / inplane_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])
            else:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / inplane_spac, 4 / inplane_spac, 4 / z_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])

            inference_statistics = np.asarray([[im.image_filled.sum() * (4 ** 3), im.solidity, im.axis_major_length * 4,
                                                im.axis_minor_length * 4, *im.inertia_tensor_eigvals] for im, _ in
                                               get_masses(inf_4mm == 1, size_thresh)])
            inference_segmentations = [im.image_filled for im, _ in get_masses(inf_4mm == 1, size_thresh)]
            inference_locations = [im.bbox for im, _ in get_masses(inf_4mm > 0, size_thresh)]
            inference_intensity = [im.image_intensity for im, _ in get_masses(inf, size_thresh, im)]

            if len(inference_centroids) == 1:
                print(case, "has 1 kidney")
                single_kidney_flag = True
                # check if sole kidney is central, and retrieve centroid of bone-attenuating tissue 
                central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(inference_centroids, im, inf,
                                                                               inf_n.header['pixdim'][3], axes=axes)
                if central_kidney_flag:
                    kidneys = ['central']
                elif inference_centroids[0][lr] - lr_bone > 0:
                    kidneys = ['left']
                else:
                    kidneys = ['right']
                print("Sole kidney is in location {}.".format(kidneys[0]))
            else:
                if (len(inference_centroids) == 0) or (len(inference_centroids) > 2): continue
                # assert(len(inference_centroids)==2)
                single_kidney_flag = False
                if inference_centroids[0][lr] < inference_centroids[1][lr]:
                    kidneys = ['right', 'left']
                else:
                    kidneys = ['left', 'right']

            centroids, statistics = [*inference_centroids], [*inference_statistics]
            segmentations = [*inference_segmentations]
            intensities = [*inference_intensity]
            locations = [*inference_locations]
            centre = np.mean(centroids, axis=0)

            if not ((inf.shape[lr] == 512) and (inf.shape[ud] == 512)):
                print("Strange im shape:", inf.shape)
                continue

            lb_cancers = np.array([np.array(centroid) for _, centroid in get_masses(lb == 2, size_thresh / 10)])
            lb_cysts = np.array([np.array(centroid) for _, centroid in get_masses(lb == 3, size_thresh / 10)])

            canc2kid = assign_labels_2_kidneys(centroids, lb_cancers)
            cyst2kid = assign_labels_2_kidneys(centroids, lb_cysts)

            cancer_vols = np.asarray([im.area * vox_volmm for im, centroid in get_masses(lb == 2, size_thresh / 10)])
            cyst_vols = np.asarray([im.area * vox_volmm for im, centroid in get_masses(lb == 3, size_thresh / 10)])

            obj_meta = np.array(
                [[seg_2_mesh(segmentations[i], axes=axes, show=is_testing), case[:-7] + '_{}'.format(kidneys[i])] for i
                 in range(len(kidneys))], dtype=object)
            objs, names = obj_meta[:, 0], obj_meta[:, 1].astype(str)

            for i, statistic in enumerate(statistics):
                obj_name = names[i] + '.obj'
                location = locations[i]
                verts = fu.create_and_save_raw_object(rawv_p, rawo_p, objs[i], names[i])
                obj_file = gmu.smooth_object(obj_name, rawo_p)
                c, v, e = gmu.extract_object_features(obj_file, obj_name)
                feature_set = feu.generate_features(case, statistic, c, kidneys[i], i, intensities[i], is_labelled=True,
                                                    cancer_vols=cancer_vols, cyst_vols=cyst_vols, canc2kid=canc2kid,
                                                    cyst2kid=cyst2kid)
                fu.save_smooth_object_data(feature_set, c, v, e, obj_file, obj_name, cleano_p, c_p, v_p, e_p)

                csv_writer = csv.DictWriter(feature_file, fieldnames=list(feature_set.keys()))
                if ((case_index == 0) and (i == 0)) and ((not csv_exists) or overwrite): csv_writer.writeheader()
                csv_writer.writerow(feature_set)

                if is_testing:
                    xmin, ymin, zmin, xmax, ymax, zmax = location
                    verts_displaced = np.round(verts + np.array([xmin, ymin, zmin]))
                    tu.plot_obj_onlabel(verts_displaced, axes, inf_4mm)

            ########### TESTING #############
            if is_testing:
                # Printing statistics for testing 
                for i, (vol, convexity, majdim, mindim, _, _, _) in enumerate(statistics): print(
                    "{} kidney has a volume of {:.3f}cm cubed.".format(kidneys[i], vol / 1000))
                for vol, assoc in zip(cancer_vols, canc2kid): print(
                    "Cancer has a volume of {:.3f}cm cubed, and belongs to the {} kidney.".format(vol / 1000,
                                                                                                  kidneys[assoc]))
                for vol, assoc in zip(cyst_vols, cyst2kid): print(
                    "Cyst has a volume of {:.3f}cm cubed, and belongs to the {} kidney.".format(vol / 1000,
                                                                                                kidneys[assoc]))
                # Plotting images for testing
                if single_kidney_flag:
                    tu.plot_all_single_kidney(im, centre, centroids[0], kidneys[0], [ud_bone, lr_bone], axes,
                                              is_labelled=True, lb_cancers=lb_cancers, lb_cysts=lb_cysts)
                else:
                    tu.plot_all_double_kidney(im, centre, centroids, kidneys, axes, is_labelled=True,
                                              lb_cancers=lb_cancers, lb_cysts=lb_cysts)
            print()


def create_unseen_dataset(path, dataset, im_p, infnpy_p, infnii_p,
                          is_testing=False, size_thresh=200, overwrite=True):
    home = os.path.join(path, 'objects')
    save_dir = os.path.join(home, dataset)
    fu.create_folder(home), fu.create_folder(save_dir)
    rawv_p, rawo_p, cleano_p, c_p, v_p, e_p = fu.setup_save_folders(save_dir)

    cases = [case for case in os.listdir(im_p) if case.endswith('.nii.gz')]
    cases.sort()

    feature_fp = os.path.join(save_dir, 'features_unlabelled.csv')
    if overwrite:
        access = 'w'
        csv_exists = False
    else:
        access = 'a'
        csv_exists = os.path.exists(feature_fp)
    with open(feature_fp, access, newline="") as feature_file:
        for case_index, case in enumerate(cases):
            ########### LOAD DATA #############
            print(case)
            if csv_exists:
                with open(feature_fp, "r") as csv_check:
                    written_cases = [row['case'] for row in csv.DictReader(csv_check)]
                    if len(written_cases) == 0:
                        csv_exists = False
                    elif case in written_cases:
                        continue
            inf_n = nib.load(os.path.join(infnii_p, case))
            inf = nifti_2_correctarr(inf_n)
            kid_data = np.array(get_masses(inf > 0, 20), dtype=object)

            im_n = nib.load(os.path.join(im_p, case))
            inf_4mm = np.load(os.path.join(infnpy_p, case[:-7] + '.npy'), allow_pickle=True)
            im = nifti_2_correctarr(im_n)

            spacing = inf_n.header['pixdim'][1:4]
            if len(kid_data) == 0: continue
            spacing_axes = find_orientation(spacing, kid_data[:, 1], is_axes=False)
            if spacing_axes == (0, 0, 0): continue
            z_spac, inplane_spac = spacing[spacing_axes[0]], spacing[spacing_axes[1]]

            axes = find_orientation(im.shape, kid_data[:, 1], im=im)
            if axes == (0, 0, 0): continue
            axial, lr, ud = axes

            if axial == 0:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / z_spac, 4 / inplane_spac, 4 / inplane_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])
            elif axial == 1:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / inplane_spac, 4 / z_spac, 4 / inplane_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])
            else:
                inference_centroids = np.asarray(
                    [np.asarray([*centroid]) * np.array([4 / inplane_spac, 4 / inplane_spac, 4 / z_spac]) for
                     _, centroid in get_masses(inf_4mm == 1, size_thresh)])

            inference_statistics = np.asarray([[im.image_filled.sum() * (4 ** 3), im.solidity, im.axis_major_length * 4,
                                                im.axis_minor_length * 4, *im.inertia_tensor_eigvals] for im, _ in
                                               get_masses(inf_4mm == 1, size_thresh)])
            inference_segmentations = [im.image_filled for im, _ in get_masses(inf_4mm == 1, size_thresh)]
            inference_locations = [im.bbox for im, _ in get_masses(inf_4mm == 1, size_thresh)]
            inference_intensity = [im.image_intensity for im, _ in get_masses(inf, size_thresh, im)]

            if len(inference_centroids) == 1:
                print(case, "has 1 kidney")
                single_kidney_flag = True
                # check if sole kidney is central, and retrieve centroid of bone-attenuating tissue 
                central_kidney_flag, ud_bone, lr_bone = is_sole_kidney_central(inference_centroids, im, inf,
                                                                               inf_n.header['pixdim'][3], axes=axes)
                if central_kidney_flag:
                    kidneys = ['central']
                elif inference_centroids[0][lr] - lr_bone > 0:
                    kidneys = ['left']
                else:
                    kidneys = ['right']
                print("Sole kidney is in location {}.".format(kidneys[0]))
            else:
                if (len(inference_centroids) == 0): continue

                single_kidney_flag = False
                if (len(inference_centroids) > 2):
                    kidneys = np.arange(1, len(inference_centroids) + 1).astype(str).tolist()
                else:
                    if inference_centroids[0][lr] < inference_centroids[1][lr]:
                        kidneys = ['right', 'left']
                    else:
                        kidneys = ['left', 'right']

            centroids, statistics = [*inference_centroids], [*inference_statistics]
            segmentations = [*inference_segmentations]
            intensities = [*inference_intensity]
            locations = [*inference_locations]
            centre = np.mean(centroids, axis=0)

            if not ((inf.shape[lr] == 512) and (inf.shape[ud] == 512)):
                print("Strange im shape:", inf.shape)
                continue

            obj_meta = np.array(
                [[seg_2_mesh(segmentations[i], axes=axes, show=is_testing), case[:-7] + '_{}'.format(kidneys[i])] for i
                 in range(len(kidneys))], dtype=object)
            objs, names = obj_meta[:, 0], obj_meta[:, 1].astype(str)

            for i, statistic in enumerate(statistics):
                obj_name = names[i] + '.obj'
                location = locations[i]
                verts = fu.create_and_save_raw_object(rawv_p, rawo_p, objs[i], names[i])
                obj_file = gmu.smooth_object(obj_name, rawo_p)
                c, v, e = gmu.extract_object_features(obj_file, obj_name)
                feature_set = feu.generate_features(case, statistic, c, kidneys[i], i, intensities[i],
                                                    is_labelled=False)
                fu.save_smooth_object_data(feature_set, c, v, e, obj_file, obj_name, cleano_p, c_p, v_p, e_p)

                csv_writer = csv.DictWriter(feature_file, fieldnames=list(feature_set.keys()))
                if ((case_index == 0) and (i == 0)) and ((not csv_exists) or overwrite): csv_writer.writeheader()
                csv_writer.writerow(feature_set)

                if is_testing:
                    xmin, ymin, zmin, xmax, ymax, zmax = location
                    verts_displaced = np.round(verts + np.array([xmin, ymin, zmin]))
                    tu.plot_obj_onlabel(verts_displaced, axes, inf_4mm)

            ########### TESTING #############
            if is_testing:
                # Printing statistics for testing 
                for i, (vol, convexity, majdim, mindim, _, _, _) in enumerate(statistics): print(
                    "{} kidney has a volume of {:.3f}cm cubed.".format(kidneys[i], vol / 1000))
                # Plotting images for testing
                if single_kidney_flag:
                    tu.plot_all_single_kidney(im, centre, centroids[0], kidneys[0], [ud_bone, lr_bone], axes=axes,
                                              is_labelled=False)
                else:
                    tu.plot_all_double_kidney(im, centre, centroids, kidneys, axes=axes, is_labelled=False)