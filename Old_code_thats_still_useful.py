import numpy
import open3d as o3d
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 18})
import sklearn.decomposition as decomp
import numpy as np
import os
from matplotlib import cm
import random
import copy
import time
import pandas as pd
import pymeshlab as pml


def graph_statistical_removal_and_PCA_variance(nb_neighbor_upper: int, nb_neighbors_samples: int, std_dev_upper: float,
                                               std_dev_samples: int, nb_neighbor_lower: int = 1,
                                               std_dev_lower: float = .000001, filepath: str = 0,
                                               point_cloud_file: o3d.geometry.PointCloud = 0) -> None:
    """
    Graphs the variance of the point cloud along PCA components as a function of statistical outlier removal.
    :param nb_neighbor_upper: Upper bound for number of neighbors argument
    :param nb_neighbors_samples: Number of samples to test for nb_neighbors
    :param std_dev_upper: upper bound for standard deviation argument
    :param std_dev_samples: Number of samples to use for standard deviation
    :param nb_neighbor_lower: lower bound for number of neighbors argument. Default 1.
    :param std_dev_lower: Lower bound for standard deviation argument. Default .000001
    :param filepath: Filepath for point cloud to be tested. Optional.
    :param point_cloud_file: Point cloud file to be tested. Optional.
    :return:
    """
    pcd = o3d.geometry.PointCloud
    if point_cloud_file != 0:
        pcd = point_cloud_file
    elif filepath != 0:
        pcd = o3d.io.read_point_cloud(filepath)
    nb_neighbor_points = np.linspace(nb_neighbor_lower, nb_neighbor_upper, nb_neighbors_samples).astype(int)
    std_dev_points = np.linspace(std_dev_lower, std_dev_upper, std_dev_samples)
    explained_variance_ratio_data = np.empty((3, nb_neighbor_points.size, std_dev_points.size))
    explained_variance_data = np.empty((3, nb_neighbor_points.size, std_dev_points.size))
    for i in range(nb_neighbor_points.size):
        for j in range(std_dev_points.size):
            downpcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbor_points[i],
                                                          std_ratio=std_dev_points[j])
            points = np.asarray(downpcd.points)
            if points.size != 0:
                PCA = decomp.PCA(n_components=3)
                PCA.fit(points)
                for k in range(3):
                    explained_variance_ratio_data[k, i, j] = PCA.explained_variance_ratio_[k]
                    explained_variance_data[k, i, j] = PCA.explained_variance_[k]
            else:
                for k in range(3):
                    explained_variance_data[k, i, j] = 'nan'
                    explained_variance_ratio_data[k, i, j] = 'nan'
    nb_neighbor_points, std_dev_points = np.meshgrid(nb_neighbor_points, std_dev_points)
    graph_labels_ratio = ['Principal Component Highest Variance Ratio', 'Principal Component Medium Variance Ratio',
                          'Principal Component Least Variance Ratio']
    graph_labels_amount = ['Variance Highest', 'Variance Medium', 'Variance Lowest']
    fig1 = []
    ax1 = []
    fig2 = []
    ax2 = []
    for i in range(3):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig1.append(fig)
        ax1.append(ax)
        ax1[i].plot_surface(std_dev_points, nb_neighbor_points, np.transpose(explained_variance_ratio_data[i, :, :]),
                            cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax1[i].set_title(graph_labels_ratio[i])
        ax1[i].set_xlabel('Standard deviation ratio')
        ax1[i].set_ylabel('Number of neighbors')
        ax1[i].set_zlabel('Variance ratio')
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        fig2.append(fig)
        ax2.append(ax)
        ax2[i].plot_surface(std_dev_points, nb_neighbor_points, np.transpose(explained_variance_ratio_data[i, :, :]),
                            cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax2[i].set_title(graph_labels_amount[i])
        ax2[i].set_xlabel('std_dev_points')
        ax2[i].set_ylabel('# of neighbors')
        ax2[i].set_zlabel('Variance Ratio')
    plt.show()
    print(np.nanmin(explained_variance_ratio_data[2, :, :]))
    return


def graph_statistical_removal_and_average_distance(nb_neighbor_upper: int, nb_neighbors_samples: int, std_dev_upper: float,
                                                   std_dev_samples: int, nb_neighbor_lower: int = 1,
                                                   std_dev_lower: float = .000001, filepath: str = 0,
                                                   point_cloud_file: o3d.geometry.PointCloud = 0) -> None:
    """
    Graphs the average distance between points as a function of statistical outlier removal.
    :param nb_neighbor_upper: Upper bound for number of neighbors argument
    :param nb_neighbors_samples: Number of samples to test for nb_neighbors
    :param std_dev_upper: upper bound for standard deviation argument
    :param std_dev_samples: Number of samples to use for standard deviation
    :param nb_neighbor_lower: lower bound for number of neighbors argument. Default 1.
    :param std_dev_lower: Lower bound for standard deviation argument. Default .000001
    :param filepath: Filepath for point cloud to be tested. Optional.
    :param point_cloud_file: Point cloud file to be tested. Optional.
    :return:
    """
    pcd = o3d.geometry.PointCloud
    if point_cloud_file != 0:
        pcd = point_cloud_file
    elif filepath != 0:
        pcd = o3d.io.read_point_cloud(filepath)
    nb_neighbor_points = np.linspace(nb_neighbor_lower, nb_neighbor_upper, nb_neighbors_samples).astype(int)
    std_dev_points = np.linspace(std_dev_lower, std_dev_upper, std_dev_samples)
    average_distance_data = np.empty((nb_neighbor_points.size, std_dev_points.size))
    for i in range(nb_neighbor_points.size):
        for j in range(std_dev_points.size):
            downpcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbor_points[i], std_ratio=std_dev_points[j])
            points = np.asarray(downpcd.points)
            if points.size != 0:
                mean_distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(downpcd))
                average_distance_data[i, j] = mean_distance
            else:
                average_distance_data[i, j] = 'nan'
    nb_neighbor_points, std_dev_points = np.meshgrid(nb_neighbor_points, std_dev_points)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(nb_neighbor_points, std_dev_points, np.transpose(average_distance_data),
                        cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_title('Average Distance Between Points')
    ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('Standard deviation')
    ax.set_zlabel('Average distance')
    plt.show()
    return


def display_noise_not_noise(cloud, ind):
    noise_cloud = cloud.select_by_index(ind)
    correct_cloud = cloud.select_by_index(ind, invert=True)

    noise_cloud.paint_uniform_color([1, 0, 0])
    correct_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([noise_cloud, correct_cloud])


def generate_gaussian_noise_pt_cloud(mean: float, std_dev_distance_mult: float, noise_percent: float, point_cloud_file: o3d.geometry.PointCloud = 0,
                                     point_cloud_loc: str = 0) -> o3d.geometry.PointCloud:
    """
    Generates Gaussian distributed noise for a point cloud. Does this by taking in a clean point, generating noise on
    only the "axis" axis, and then adding the two point clouds together.
    :param noise_percent: what percentage of points are moved from their original location.
    :param mean: mean distance the noise moves from the original point.
    :param std_dev_distance_mult: Standard deviation of the original points to the noise generated. Measured by taking
    this argument and multiplying it by the average distance between points in the point cloud.
    :param point_cloud_file: Point cloud data type, optional.
    :param point_cloud_loc: Location of point cloud file, optional.
    :return: Returns a point cloud with Gaussian noise added.
    """
    indeces = []
    points = np.array([1, 1])
    pcd = o3d.geometry.PointCloud()
    if point_cloud_file != 0:
        points = np.asarray(point_cloud_file.points)
        pcd = point_cloud_file
    elif point_cloud_loc != 0:
        point_cloud_in = o3d.io.read_point_cloud(point_cloud_loc)
        pcd = point_cloud_in
        points = np.asarray(point_cloud_in.points)
    distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcd))
    std_dev = distance*std_dev_distance_mult
    for i in range(len(points)):
        if random.random() <= noise_percent:
            indeces.append(i)
            noise = np.random.normal(mean, std_dev, (1, 3))
            points[i] = points[i] + noise
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(points)
    display_noise_not_noise(new_cloud, indeces)
    return new_cloud


def generate_point_cloud_from_mesh(number_of_points: int, mesh_file: o3d.geometry.TriangleMesh = 0, mesh_loc: str = 0,
                                   ) -> o3d.geometry.PointCloud:
    """
    Generate a point cloud from a mesh using uniformly sampled points across the mesh.
    :param number_of_points: Number of points to be uniformly sampled.
    :param mesh_file: Actual triangle mesh to sample. Optional.
    :param mesh_loc: Location of the file to sample. Optional.
    :return: Return a point cloud created from this input mesh.
    """
    triangleMesh = o3d.geometry.TriangleMesh()
    if mesh_loc != 0:
        triangleMesh = o3d.io.read_triangle_mesh(mesh_loc)
    elif mesh_file != 0:
        triangleMesh = mesh_file
    generated_cloud = o3d.geometry.TriangleMesh.sample_points_uniformly(triangleMesh, number_of_points=number_of_points)
    return generated_cloud


def evaluate_filter_parameters(nb_neighbors: int, std_ratio: float, point_cloud: o3d.geometry.PointCloud) -> None:
    """
    Evaluate the filter quality for the statistical outlier removal filter. Reports the total points removed and the
    total noisy points removed. Note: MVP point clouds always start with 2048 points.
    :param nb_neighbors: nb_neighbors argument to use for the statistical outlier removal filter
    :param std_ratio: std_ratio argument to use for the statistical outlier removal filter
    :param point_cloud: Point cloud file to test
    :return: Nothing, prints the quality metrics.
    """
    pcd_in = copy.copy(point_cloud)
    o3d.visualization.draw_geometries([pcd_in])
    pcd_noisy = generate_gaussian_noise_pt_cloud(0, 2, .2, point_cloud_file=pcd_in)
    pcd_in = copy.copy(point_cloud)
    clean_array = np.asarray(pcd_in.points)
    noisy_array = np.asarray(pcd_noisy.points)
    noise_point_original = 0
    for point in range(np.size(clean_array, 0)):
        if (noisy_array[point] - clean_array[point] != [0, 0, 0]).any():
            noise_point_original += 1
    pcd_cleaned, ind = pcd_noisy.remove_statistical_outlier(nb_neighbors, std_ratio)
    o3d.visualization.draw_geometries([pcd_cleaned])
    noise_point_new = 0
    for point in range(np.size(clean_array, 0)):
        if ind.count(point) == 1:
            if (noisy_array[point] - clean_array[point] != [0, 0, 0]).any():
                noise_point_new += 1

    noise_points_removed = noise_point_original - noise_point_new
    total_points_removed = 2048 - len(ind)

    print('noise points removed: ' + str(noise_points_removed))
    print('total points removed: ' + str(total_points_removed))
    return


def voxel_downsample(voxel_lower_bound: int, voxel_upper_bound: int, samples: int, point_cloud: o3d.geometry.PointCloud = 0, point_cloud_path: str = 0) -> int:
    """
    Downsample point cloud at various voxel sizes. Creates a data table displaying voxel size, average distance between
    points and their nearest neighbor, meshability using rolling ball approach, and size of resulting mesh, if it meshes.
    :param voxel_lower_bound: Lower bound for the voxel size graphed. Default is 0.
    :param voxel_upper_bound: Upper bound for the voxel size graphed.
    :param samples: step size for the voxel sizes used for the graph.
    :param point_cloud: point cloud to be tested.
    :param point_cloud_path: file path for cloud to be tested
    :return: Outputs a data table. Returns a 1 if the function executes successfully.
    """
    data = pd.DataFrame(columns=['Voxel Size', 'Average Distance Between Points', 'Meshable', 'Mesh Size (bytes)'])
    
    voxel_sizes = np.linspace(voxel_lower_bound, voxel_upper_bound, samples)
    for voxel_size in voxel_sizes:
        pcd = o3d.geometry.PointCloud()
        if point_cloud != 0:
            pcd = point_cloud
        elif point_cloud_path != 0:
            pcd = o3d.io.read_point_cloud(point_cloud_path)
        pcd_downsampled = pcd.voxel_down_sample(voxel_size)
        average_distance = str(np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcd_downsampled)))
        pmlMeshSet = pml.MeshSet()
        pmlMeshSet.load_new_mesh(file_name=pcd_downsampled)
        pmlMeshSet.surface_reconstruction_ball_pivoting()
        meshable = pmlMeshSet.is_meshable()
        if pmlMeshSet.is_meshable():
            mesh_size = pmlMeshSet.mesh().size()
        print('\n')
