import open3d as o3d
import Point_cloud_class as pcc
import glob
import os
import random
import time

def view_point_cloud(point_cloud):
    o3d.visualization.draw_geometries([point_cloud])
    
def main(clouds_to_select, runs_per_cloud, input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper, nb_neighbors_samples, std_dev_upper, std_dev_samples, nb_neighbors_lower, std_dev_lower, folder_path):
        pointCloudCollection = pcc.PointCloudDatabase(input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper,
                                                      nb_neighbors_samples, std_dev_upper, std_dev_samples,
                                                          nb_neighbors_lower, std_dev_lower)
        pointCloudCollection.record_point_cloud_information(folder_path=folder_path, runs_per_cloud=runs_per_cloud, clouds_to_select=clouds_to_select)


if __name__ == "__main__":
    
    # #point_cloud_files = glob.glob(os.path.join(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', '*.ply'))
    
    # # point_cloud_files = list(point_cloud_files)
    # # random.shuffle(point_cloud_files)
    
    # # point_cloud_files = [point_cloud for point_cloud in point_cloud_files if 'pcd116.ply' in point_cloud]
    
    # for i in range(0,2400,50):
         point_cloud = pcc.PointCloudStructure(pointCloudFilePath=f'C:/Users/Michael/Desktop/VRAC/Boeing Digital Twin/Journal Article/Code/PointCloudFiltering/3D_files/MVP Point Clouds/pcd2100.ply', show=True)
    #     print(f'pcd{i}')
    
    # point_cloud.clean_noisy_cloud(100, 3, show=True)
    
    # point_cloud.clean_noisy_cloud(5, .000001, show=True)
    
    # # view_point_cloud(point_cloud)
    
    # # for point_cloud_file in point_cloud_files:
        
    # #     print(point_cloud_file)
    
        # torch_point_cloud = pcc.PointCloudStructure(pointCloudFilePath=r"C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\torch\torch\pset-torch-lfd-unfiltered-s13.ply", show=True)
         
        # statue_point_cloud = pcc.PointCloudStructure(pointCloudFilePath=r"C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\statue\statue\pset-statue-mve-unfiltered.ply", show=True)
         
        #  torch_point_cloud.insert_custom_noisy_point_cloud(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\torch\torch\pset-torch-lfd-unfiltered-s13.ply')
         
        #  statue_point_cloud.insert_custom_noisy_point_cloud(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\statue\statue\pset-statue-mve-unfiltered.ply')
         
        #  print("Number of points in original statue cloud", len(statue_point_cloud.noisy_cloud.points))
         
        #  print("Number of points in original torch cloud", len(torch_point_cloud.noisy_cloud.points))
         
        #  print("Number of points in wolff's filtered torch cloud", len(torch_point_cloud.PointCloud.points))
         
        #  print("Number of points in wolff's filtered statue cloud", len(statue_point_cloud.PointCloud.points))
         
        #  torch_point_cloud.generate_statistical_removal_and_average_distance_data(100, 10, 3, 10, 1, .00001, graph=False)
         
        #  print("Done with torch average distance")
         
        #  statue_point_cloud.generate_statistical_removal_and_average_distance_data(100, 10, 3, 10, 1, .00001, graph=False)
         
        #  print("Done with statue average distance")
         
        #  torch_point_cloud.generate_statistical_removal_and_PCA_variance_data(100, 10, 3, 10, 1, .00001, graph=False)
         
        #  print("Done with torch PCA")
         
        #  statue_point_cloud.generate_statistical_removal_and_PCA_variance_data(100, 10, 3, 10, 1, .00001, graph=False)
         
        #print("Done with statue PCA")
         
        #  torch_point_cloud.clean_noisy_cloud(method='Average distance', show=True)
         
        #  statue_point_cloud.clean_noisy_cloud(method='Average distance', show=True)
         
        #  number_of_statue_points_after_average_distance = len(statue_point_cloud.cleaned_cloud.points)
         
        #  number_of_torch_points_after_average_distance = len(torch_point_cloud.cleaned_cloud.points)
         
        #  print("Number of points in average distance filtered statue cloud", number_of_statue_points_after_average_distance)
         
        #  print("Number of points in average distance filtered torch cloud", number_of_torch_points_after_average_distance)
         
        #  torch_point_cloud.clean_noisy_cloud(method='PCA', show=True)
         
        #  statue_point_cloud.clean_noisy_cloud(method='PCA', show=True)
         
        #  number_of_statue_points_after_PCA = len(statue_point_cloud.cleaned_cloud.points)
         
        #  number_of_torch_points_after_PCA = len(torch_point_cloud.cleaned_cloud.points)
         
        #  print("Number of points in PCA filtered statue cloud", number_of_statue_points_after_PCA)
         
        #  print("Number of points in PCA filtered torch cloud", number_of_torch_points_after_PCA)
        
        # random.seed(0)
        
        # start_time = time.time()
         
        # main(10, 1, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 10, 3, 10, 1, .00001, 'Test2')
         
        # run_time = time.time() - start_time
        
        # print("Run time 1: ", run_time) 
    
        #  point_cloud = pcc.PointCloudStructure(pointCloudFilePath=r"C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\torch\torch\pset-torch-acts-filtered.ply", show=True)
        
        #  point_cloud.generate_gaussian_noise(0, .05, .2, True)
         
        # #  point_cloud.generate_statistical_removal_and_average_distance_data(100, 32, 3, 32, 1, .00001, graph=True)
    
        # # #  point_cloud.generate_statistical_removal_and_PCA_variance_data(100, 32, 3, 32, 1, .00001, graph=False)
        
        # #  point_cloud.clean_noisy_cloud(method='Average distance', show=True)
         
        # #  point_cloud.clean_noisy_cloud(500, 4, show=True)
    
        # # point_cloud.clean_noisy_cloud(method='PCA', show=True)
        
        

        #  point_cloud.insert_custom_noisy_point_cloud(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\torch\torch\pset-torch-ps-unfiltered.ply')

        #  point_cloud.generate_statistical_removal_and_average_distance_data(100, 10, 3, 10, 1, .00001, graph=False)
    
        #  point_cloud.generate_statistical_removal_and_PCA_variance_data(100, 10, 3, 10, 1, .00001, graph=False)

        # #  point_cloud.clean_noisy_cloud(50, .000001, show=True)

        #  point_cloud.clean_noisy_cloud(method='Average distance', show=True)
    
        #  point_cloud.clean_noisy_cloud(method='PCA', show=True)
    
        #  number_of_original_points = len(original_point_cloud.PointCloud.points)
    
        #  number_of_filtered_points_PCA = len(point_cloud.best_PCA_cleaned_cloud.points)
    
        #  number_of_filtered_points_average_distance = len(point_cloud.best_average_distance_cleaned_cloud.points)
    
        #  print("Number of original points: ", number_of_original_points)
    
        #  print("Number of filtered points PCA: ", number_of_filtered_points_PCA)
    
        #  print("Number of filtered points average distance: ", number_of_filtered_points_average_distance)
        
        #  print("Number of points in wolff's filtered cloud", len(point_cloud.PointCloud.points))
         
        #  print(len(point_cloud.cleaned_cloud.points))
    
    
    
