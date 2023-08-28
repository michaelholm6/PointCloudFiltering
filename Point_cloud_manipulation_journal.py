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


if __name__ == "__main__":
    pcd_test = o3d.io.read_point_cloud(filename='Various Mesh Files/Input/MVP Point Clouds/pcd2.pcd')
    graph_statistical_removal_and_PCA_variance(2, 100, .0001, point_cloud_file=pcd_test)
