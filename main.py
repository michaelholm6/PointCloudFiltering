import numpy as np
import open3d as o3d

original_point_cloud = "Various Mesh Files/Original_point_cloud.ply"

if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud(filename=original_point_cloud)
    """
    Reads in "Original point cloud" from project fold as a point cloud file.
    """

    downpcd = pcd.voxel_down_sample(voxel_size=.05)
    """
    Down sample the point cloud by taking voxels of side length "voxel_size" and averaging the position, color, 
    and normals of all points in this voxel. All points in this voxel are deleted, and replaced with a single point.
    Returns a filtered point cloud.
    """

    cl, ind = downpcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=1)
    """
    Remove statistical outliers by taking the average distance to the nearest "nb_neighbors" points, and comparing this
    to the general average of the point cloud. If the average distance lies more than "std_ratio" standard deviations
    apart, the point is deleted. Returns a tuple of filtered point cloud and index of all points removed.
    """

    cl, ind = cl.remove_radius_outlier(nb_points=5, radius=1)
    """
    Remove points that have less than "nb_points" in a sphere of radius "radius" around them. Returns a tuple of a 
    filtered point cloud and the index of all points that are removed.
    """

    o3d.visualization.draw_geometries([cl])

    o3d.io.write_point_cloud(filename="Various Mesh Files/filtered_point_cloud.ply", pointcloud=cl)
    """
    Save filtered point cloud to the project folder as a point cloud.
    """

    distances = cl.compute_nearest_neighbor_distance()
    """
    Computes the distance to the nearest neighbor for every point of the point cloud. Return a double vector, which is 
    an Open3D class that is basically just a one dimensional array, but is special to Open3D.
    """

    avg_dist = np.mean(distances)
    """
    Calculates the average distance for each point to its closest neighbor. Returns an int.
    """

    radius = avg_dist
    """
    Size of the radius of the sphere used for the create_point_cloud_ball_pivoting function later in this script.
    Smaller sphere means more accurate creation, but also more triangles.
    """

    o3d.geometry.PointCloud.estimate_normals(self=cl)
    """
    Estimates normals for the cl point cloud, by using the input point cloud. Some linear algebra stuff that I don't
    understand. 
    """

    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd=cl, radii=o3d.utility.DoubleVector([radius*.8]))
    """
    Create a mesh by using F. Bernardini et al., “The ball-pivoting algorithm for surface reconstruction”, 1999 
    workflow, which uses a ball of radius "radius" to form triangles. The ball is moved around the point cloud, and
    anytime the ball touches three points, without touching any other points, a triangle is created. This is repeated
    for all radius arguments in the DoubleVector argument.
    """

    o3d.visualization.draw_geometries([bpa_mesh])
    o3d.io.write_triangle_mesh(filename="Various Mesh Files/test.stl", mesh=bpa_mesh)
    """
    Save the triangle mesh to the project folder in STL format
    """

    bpa_mesh = o3d.geometry.TriangleMesh.filter_smooth_simple(self=bpa_mesh, number_of_iterations=3)
    """
    Smooths the triangle mesh, using "number_of_iterations" for the number of times the filter is applied. Simple
    neighbor average used, with this equation: 
    vo=vi+∑n∈Nvn)/|N|+1, with vi being the input value, vo the output value, and N is the set of adjacent neighbours.
    """

    bpa_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(self=bpa_mesh)
    """
    Compute normals for the triangle mesh. Compares the triangle mesh to the input point cloud.
    """

    o3d.visualization.draw_geometries([bpa_mesh])
    """
    Displays the created triangle mesh. Waits for the user to X out of the displayed image.
    """

    o3d.io.write_triangle_mesh(filename="Various Mesh Files/test_smoothed.stl", mesh=bpa_mesh)
    """
    Saves the triangle mesh to the project folder as an STL, after being smoothed. 
    """
