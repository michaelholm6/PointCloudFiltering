import numpy
import open3d as o3d
import matplotlib.pyplot as plt
import sklearn.decomposition as decomp
import numpy as np
import os
from matplotlib import cm
import random

original_point_cloud = "Various Mesh Files/filtered_point_cloud.ply"


def graph_voxel_down_sample_and_file_size(voxel_size_upper: float, sample_points: int, voxel_size_lower: float = .0001,
                                          filepath: str = 0, point_cloud_file: o3d.geometry.PointCloud = 0) -> None:
    """
    Graph the relationship of voxel down sample size and corresponding file size in bytes.
    :param voxel_size_upper: Upper bound on voxel size to graph.
    :param sample_points: number of samples for the graph.
    :param voxel_size_lower: Lower bound of voxel size to graph. Default 0.
    :param filepath: filepath of point cloud to analyze, optional.
    :param point_cloud_file: point cloud to analyze, optional.
    :return: Graphs the file size and average distance between points as a function of voxel size
    """
    pcd = o3d.geometry.PointCloud
    if filepath != 0:
        pcd = o3d.io.read_point_cloud(filename=filepath)
    elif point_cloud_file != 0:
        pcd = point_cloud_file
    x = np.linspace(voxel_size_lower, voxel_size_upper, sample_points)
    y1 = []
    y2 = []
    for i in x:
        downpcd = pcd.voxel_down_sample(voxel_size=i)
        o3d.io.write_point_cloud('Various Mesh Files/temp.ply', downpcd)
        size = os.path.getsize('Various Mesh Files/temp.ply')
        distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(downpcd))
        y1.append(size)
        y2.append(distance)
    os.remove('Various Mesh Files/temp.ply')
    plt.subplot(1, 2, 1)
    plt.plot(x, y1)
    plt.ylabel('File size in bytes')
    plt.xlabel('Voxel Size')
    plt.subplot(1, 2, 2)
    plt.plot(x, y2)
    plt.xlabel('voxel Size')
    plt.ylabel('Average distance between points')
    plt.show()


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
            downpcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbor_points[i], std_ratio=std_dev_points[j])
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
    graph_labels_ratio = ['Variance Ratio Highest', 'Variance Ratio Medium', 'Variance Ratio Lowest']
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
        ax1[i].set_xlabel('Standard deviation')
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
    ax.set_xlabel('Number of neighbors')
    ax.set_ylabel('Standard deviation')
    ax.set_zlabel('Average distance')
    plt.show()


def generate_gaussian_noise_pt_cloud_deprecated(mean: float, std_dev: float, point_cloud_file: o3d.geometry.PointCloud = 0,
                                     point_cloud_loc: str = 0, axis: int = 2) -> o3d.geometry.PointCloud:
    """
    Generates Gaussian distributed noise for a point cloud. Does this by taking in a clean point, generating noise on
    only the "axis" axis, and then adding the two point clouds together.
    :param mean: mean distance the noise moves from the original point.
    :param std_dev: Standard deviation of the original points to the noise generated.
    :param axis: 0 = noise on x axis, 1 = noise on y axis, 2 = noise on z axis.
    :param point_cloud_file: Point cloud data type, optional.
    :param point_cloud_loc: Location of point cloud file, optional.
    :return: Returns a point cloud with Gaussian noise added.
    """
    points = np.array([1, 1])
    if point_cloud_file != 0:
        points = np.shape(np.asarray(point_cloud_file.points))
    elif point_cloud_loc != 0:
        point_cloud_in = o3d.io.read_point_cloud(point_cloud_loc)
        points = np.shape(np.asarray(point_cloud_in.points))
    noise = np.random.normal(mean, std_dev, [points[0], 1])
    noise = np.append(noise, np.zeros([points[0], 2]), axis=1)
    if axis == 0:
        noise[:, [0, 1, 2]] = noise[:, [0, 1, 2]]
    elif axis == 1:
        noise[:, [0, 1, 2]] = noise[:, [1, 0, 2]]
    elif axis == 2:
        noise[:, [0, 1, 2]] = noise[:, [1, 2, 0]]
    original_points = np.asarray(point_cloud_file.points)
    noisy_points = o3d.geometry.PointCloud()
    noisy_points.points = o3d.utility.Vector3dVector(np.asarray(point_cloud_file.points) + noise)
    noisy_points = np.asarray(o3d.utility.Vector3dVector(np.asarray(point_cloud_file.points) + noise))
    new_cloud_points = np.concatenate((original_points, noisy_points), axis=0)
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(new_cloud_points)
    return new_cloud


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
            noise = np.random.normal(mean, std_dev, (1, 3))
            points[i] = points[i] + noise
    new_cloud = o3d.geometry.PointCloud()
    new_cloud.points = o3d.utility.Vector3dVector(points)
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


def graph_point_distance(filename: str = 0, point_cloud_file: o3d.geometry.PointCloud = 0) -> None:
    if filename != 0:
        pcd = o3d.io.read_point_cloud(filename)
    elif point_cloud_file != 0:
        pcd = point_cloud_file
    distances = pcd.compute_nearest_neighbor_distance()
    plt.hist(distances, bins=5000)
    plt.show()


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud(filename='Various Mesh Files/Input/Missile Rail Point Clouds/R13.ply')
    o3d.visualization.draw_geometries([pcd])
    #graph_voxel_down_sample_and_file_size(2, 100, point_cloud_file=pcd)
    #pcd = pcd.voxel_down_sample(voxel_size=.122)
    #o3d.visualization.draw_geometries([pcd])
    #pcd = generate_gaussian_noise_pt_cloud(0, 2, .2, point_cloud_file=pcd)
    #o3d.visualization.draw_geometries([pcd])
    #graph_statistical_removal_and_PCA_variance(50, 30, 3, 5, 1, point_cloud_file=pcd)
    pcd, ind = pcd.remove_statistical_outlier(10, .0001)
    o3d.visualization.draw_geometries([pcd])
    #graph_statistical_removal_and_average_distance(50, 20, 3, 3, 1, std_dev_lower=.001, point_cloud_file=pcd)