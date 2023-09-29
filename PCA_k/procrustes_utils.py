import os
import time

import numpy as np
import open3d as o3d
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from tqdm import tqdm


# write function to read vertices from obj file
def load_obj_file(file_path):
    """
    Load an OBJ file and return vertices and faces.

    Parameters:
    - file_path: Path to the OBJ file

    Returns:
    - vertices: List of vertex coordinates
    - faces: List of faces represented as vertex indices
    """
    vertices = []
    faces = []

    try:
        with open(file_path, 'r') as obj_file:
            for line in obj_file:
                parts = line.strip().split()
                if not parts:
                    continue

                if parts[0] == 'v':
                    # Vertex definition
                    vertex = tuple(map(float, parts[1:]))
                    vertices.append(vertex)
                elif parts[0] == 'f':
                    # Face definition
                    face = [int(vertex.split('/')[0]) - 1 for vertex in parts[1:]]
                    if len(face) >= 3:
                        faces.append(face)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return vertices, faces


def convert_rectangular_to_triangular(vertices, faces):
    """
    Convert a rectangular 3D mesh into a triangular 3D mesh by splitting all rectangular faces.

    Parameters:
    - vertices: List of vertex coordinates
    - faces: List of faces represented as vertex indices

    Returns:
    - new_faces: List of triangular faces
    """
    new_faces = []

    for face in faces:
        if len(face) == 4:
            # Split the rectangular face into two triangles
            triangle1 = [face[0], face[1], face[2]]
            triangle2 = [face[2], face[3], face[0]]
            new_faces.append(triangle1)
            new_faces.append(triangle2)
        else:
            # Keep existing triangular faces unchanged
            new_faces.append(face)

    return new_faces


def process_dataframe(df, obj_folder, number_of_points=1000):
    pointclouds = []

    for index, row in df.iterrows():
        obj_fn = row['case'][:-7] + '_' + row['position'] + '.obj'
        obj_fp = os.path.join(obj_folder, obj_fn)
        vertices, simplices = load_obj_file(obj_fp)
        simplices = convert_rectangular_to_triangular(vertices, simplices)
        vertices = np.array(vertices)
        simplices = np.array(simplices)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(simplices)
        pcd = mesh.sample_points_poisson_disk(number_of_points=number_of_points)
        pointclouds.append(pcd)

    return pointclouds


def procrustes_analysis(target_points, reference_pointclouds, include_target=True,
                        max_iterations=20000, tolerance=1e-7):
    aligned_pointclouds = []

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
    if include_target:
        aligned_pointclouds.append(np.asarray(target_cloud.points))

    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations,
                                                                     relative_fitness=tolerance,
                                                                     relative_rmse=tolerance)

    for i in tqdm(range(len(reference_pointclouds))):
        source_cloud = reference_pointclouds[i]
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud,
            max_correspondence_distance=1000,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=icp_criteria
        )
        source_cloud.transform(reg_p2p.transformation)
        pc = np.asarray(source_cloud.points)
        distances = cdist(target_cloud.points, pc)
        target_index, source_index = linear_sum_assignment(distances)
        aligned_pointclouds.append(np.asarray(source_cloud.points)[source_index]) # ensures common-index nodes have the same anatomical meaning

    aligned_pointclouds = np.array(aligned_pointclouds)
    average_pointcloud = np.mean(aligned_pointclouds, axis=0)

    return average_pointcloud, aligned_pointclouds


def procrustes_analysis_normalised(target_points, reference_pointclouds, include_target=True,
                                   max_iterations=20000, tolerance=1e-7):
    aligned_pointclouds = []

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target_points)
    if include_target:
        aligned_pointclouds.append(np.asarray(target_cloud.points))

    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations,
                                                                     relative_fitness=tolerance,
                                                                     relative_rmse=tolerance)
    time.sleep(1)
    for i in tqdm(range(len(reference_pointclouds))):
        source_cloud = reference_pointclouds[i]
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_cloud, target_cloud,
            max_correspondence_distance=1000,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=icp_criteria
        )
        source_cloud.transform(reg_p2p.transformation)
        pc = np.asarray(source_cloud.points)
        distances = cdist(target_cloud.points, pc)
        target_index, source_index = linear_sum_assignment(distances)
        aligned_pointclouds.append(np.asarray(source_cloud.points)[source_index])

    aligned_pointclouds = np.array(aligned_pointclouds)
    aligned_pointclouds = np.array([pc / cdist(pc, pc).max() for pc in aligned_pointclouds])
    average_pointcloud = np.mean(aligned_pointclouds, axis=0)

    return average_pointcloud, aligned_pointclouds


# Main function
def find_average(df, obj_folder, number_of_points, n_iter, tolerance):
    pointclouds = process_dataframe(df, obj_folder, number_of_points=number_of_points)

    target_pointcloud = np.asarray(pointclouds[0].points)
    target_pointcloud -= np.mean(target_pointcloud, axis=0)
    print('Pass one of two')

    average_pointcloud, _ = procrustes_analysis(target_pointcloud, pointclouds[1:], include_target=True,
                                                max_iterations=n_iter, tolerance=tolerance)
    average_pointcloud -= np.mean(average_pointcloud, axis=0)
    print('Pass two of two')
    return procrustes_analysis(average_pointcloud, pointclouds, include_target=False, max_iterations=n_iter,
                               tolerance=tolerance)


def find_average_normalised(df, obj_folder, number_of_points, n_iter, tolerance):
    pointclouds = process_dataframe(df, obj_folder, number_of_points=number_of_points)

    target_pointcloud = pointclouds[0].points
    target_pointcloud -= np.mean(target_pointcloud, axis=0)
    target_pointcloud /= cdist(target_pointcloud, target_pointcloud).max()

    print('Pass one of two')
    average_pointcloud, _ = procrustes_analysis_normalised(target_pointcloud, pointclouds[1:], include_target=True,
                                                           max_iterations=n_iter, tolerance=tolerance)
    average_pointcloud -= np.mean(average_pointcloud, axis=0)
    average_pointcloud /= cdist(average_pointcloud, average_pointcloud).max()

    print('Pass two of two')
    return procrustes_analysis_normalised(average_pointcloud, pointclouds, include_target=False, max_iterations=n_iter,
                                          tolerance=tolerance)
