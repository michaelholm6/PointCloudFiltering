import open3d as o3d
import Point_cloud_statistics_IITSEC as pcs
import bpy
import pymeshlab
import pymeshlab as pml

original_point_cloud = "Various Mesh Files/Input/RGB&3DpointClouds/MissilesRailScanningPointCloud/R5.ply"

if __name__ == "__main__":

    pcd = o3d.io.read_point_cloud(filename=original_point_cloud)
    """
    Reads in "Original point cloud" from project folder as a point cloud file.
    """

    o3d.visualization.draw_geometries([pcd])

    #pcs.graph_statistical_removal_and_PCA_variance(100, 50, 5, 5, point_cloud_file=pcd)


    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=2, std_ratio=1)
    """
    Remove statistical outliers by taking the average distance to the nearest "nb_neighbors" points, and comparing this
    to the general average of the point cloud. If the average distance lies more than "std_ratio" standard deviations
    apart, the point is deleted. Returns a tuple of filtered point cloud and index of all points removed.
    """

    o3d.visualization.draw_geometries([cl])


    #pcs.graph_voxel_down_sample_and_file_size(50, 1000, point_cloud_file=cl)

    cl = cl.voxel_down_sample(2)
    #cl, ind = cl.remove_radius_outlier(nb_points=5, radius=1)
    """
    Remove points that have less than "nb_points" in a sphere of radius "radius" around them. Returns a tuple of a 
    filtered point cloud and the index of all points that are removed.
    """

    o3d.visualization.draw_geometries([cl])


    o3d.io.write_point_cloud(filename="Various Mesh Files/Output/filtered_point_cloud.ply", pointcloud=cl)
    """
    Save filtered point cloud to the project folder as a point cloud.
    """

    pmlMeshSet = pml.MeshSet()
    """
    Creates MeshLab instance to be operated on.
    """

    pmlMeshSet.load_new_mesh(file_name='Various Mesh Files/Output/filtered_point_cloud.PLY')
    """
    Loads PLY file into the MeshLab file.
    """

    pmlMeshSet.surface_reconstruction_ball_pivoting()
    """
    Create a mesh by using F. Bernardini et al., “The ball-pivoting algorithm for surface reconstruction”, 1999
    workflow, which uses a ball of default radius, defined by MeshLab, to form triangles. The ball is moved around the point cloud, and
    anytime the ball touches three points, without touching any other points, a triangle is created. 
    """

    pmlMeshSet.close_holes(maxholesize=100000)
    """
    Closes all holes with "maxholesize" edges or less
    """


    pmlMeshSet.remove_isolated_pieces_wrt_face_num(mincomponentsize=700)
    """
    Removes isolated components with less than 1200 faces.
    """

    pmlMeshSet.save_current_mesh('Various Mesh Files/Output/cleaned_mesh.PLY')
    """
    Saving cleaned mesh to pass to Open3D for smoothing.
    """

    bpa_mesh = o3d.io.read_triangle_mesh('Various Mesh Files/Output/cleaned_mesh.ply')
    """
    Reading in cleaned PyMeshLab mesh.
    """

    bpa_mesh = o3d.geometry.TriangleMesh.filter_smooth_simple(self=bpa_mesh, number_of_iterations=3)
    """
    Smooths the triangle mesh, using "number_of_iterations" for the number of times the filter is applied. Simple
    neighbor average used, with this equation:
    vo=vi+∑n∈Nvn)/|N|+1, with vi being the input value, vo the output value, and N is the set of adjacent neighbours.
    """

    o3d.visualization.draw_geometries([bpa_mesh])
    o3d.io.write_triangle_mesh("Various Mesh Files/Output/smoothed_cleaned_mesh.ply", bpa_mesh)
    """
    Saves smoothed Open3D mesh for passing to PyMeshLab for face reduction.
    """

    pmlMeshSet.load_new_mesh(file_name='Various Mesh Files/Output/smoothed_cleaned_mesh.ply')
    """
    Reading in smoothed mesh
    """

    pmlMeshSet.simplification_quadric_edge_collapse_decimation(targetfacenum=100000, preserveboundary=True, boundaryweight=5)
    """
    Reduces file to have "targetfacenum" faces. "Boundaryweight" defines how important boundary preservation is. Default is 1.
    """

    pmlMeshSet.save_current_mesh('Various Mesh Files/Output/reduced_mesh.ply')
    """
    Saves reduced mesh for passing to Blender.
    """

    bpy.ops.import_mesh.ply(filepath='Various Mesh Files/Output/reduced_mesh.ply')
    """
    Passing reduced_mesh into Blender instance. 
    """

    bpy.ops.object.modifier_add(type='SOLIDIFY')
    """
    Adds the solidify modifier. This modifier will convert the mesh from a surface to a solid. Blender automatically 
    calculates normals, and does it very well. 
    """

    bpy.ops.object.modifier_apply(modifier="Solidify")
    """
    Applies the above solidify modifier to the imported object.
    """

    bpy.ops.object.select_all(action='DESELECT')
    """
    Deselects all objects in the scene. There is currently a default cube and the imported mesh in this scene.
    """

    bpy.data.objects['Cube'].select_set(True)
    """
    Selects the default cube that's still in the scene.
    """

    bpy.ops.object.delete(use_global=False, confirm=False)
    """
    Deletes only the selected object. "use-global" makes it delete only the selected object, and "confirm" set to false
    suppresses any GUI from popping up. 
    """

    bpy.ops.object.select_all(action='DESELECT')
    """
    Deselects all objects in the scene.
    """

    bpy.data.objects['reduced_mesh'].select_set(True)
    """
    Selects the imported mesh.
    """

    bpy.ops.export_mesh.ply(filepath="Various Mesh Files/Output/blender_thickened.ply", use_selection=True)
    """
    Exports the thickened mesh as a PLY file. "use_selections" causes the export to only export the selected object.
    """

    bpa_mesh = o3d.io.read_triangle_mesh('Various Mesh Files/Output/blender_thickened.ply')
    """
    Reads in exported Blender mesh for visualization. 
    """

    bpa_mesh = o3d.geometry.TriangleMesh.compute_triangle_normals(self=bpa_mesh)
    """
    Compute normals for the triangle mesh. Uses the normals of the input as reference.
    """

    o3d.visualization.draw_geometries([bpa_mesh])
    """
    Displays the final created triangle mesh. Waits for the user to X out of the displayed image.
    """
