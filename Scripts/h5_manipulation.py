import h5py
import open3d as o3d
import numpy as np


if __name__ == '__main__':
    
    file = h5py.File('3D_files/MVP_Train_CP.h5', 'r')
    point_clouds = file['complete_pcds']

    for ind, point_cloud in enumerate(point_clouds):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        o3d.io.write_point_cloud("3D_files/MVP Point Clouds/pcd" + str(ind) + ".ply", pcd)

