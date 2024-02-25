import Point_cloud_class as pcc
import cProfile
import pstats
import winsound

def main(clouds_to_select, runs_per_cloud, input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper, nb_neighbors_samples, std_dev_upper, std_dev_samples, nb_neighbors_lower, std_dev_lower, folder_path):
        pointCloudCollection = pcc.PointCloudDatabase(input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper,
                                                      nb_neighbors_samples, std_dev_upper, std_dev_samples,
                                                          nb_neighbors_lower, std_dev_lower)
        pointCloudCollection.record_point_cloud_information(folder_path=folder_path, runs_per_cloud=runs_per_cloud, clouds_to_select=clouds_to_select)

if __name__ == "__main__":
    
    # main(10, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 10, 3, 10, 1, .00001, '.05.20')
    # main(30, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .1, .20, 100, 10, 3, 10, 1, .00001, '.10.20')
    # main(30, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .10, 100, 10, 3, 10, 1, .00001, '.05.10')
    # main(30, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .30, 100, 10, 3, 10, 1, .00001, '.05.30')
    main(30, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .01, .20, 100, 10, 3, 10, 1, .00001, '.01.20_')
    
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 5, 3, 5, 1, .00001, 'test5x5')
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 7, 3, 7, 1, .00001, 'test7x7')
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 10, 3, 10, 1, .00001, 'test10x10')
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 14, 3, 14, 1, .00001, 'test14x14')
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 22, 3, 22, 1, .00001, 'test22x22')
    # main(r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 32, 3, 32, 1, .00001, 'test32x32')

    # cProfile.run('main()', 'my_func_stats')
    
    # stream = open('output.txt', 'w')
    # p = pstats.Stats("my_func_stats", stream=stream)
    # p.sort_stats("cumulative").print_stats()
        
    
    
    #    test_cloud = pcc.PointCloudStructure(pointCloudFilePath=r"C:\Users\Michael\Desktop\upsampled_cloud.pcd")
    #    test_cloud.generate_gaussian_noise(0, 100, .2, True)
    #    #test_cloud.generate_chamfer_distance_data(100, 10, 3, 8, 1, .00001, True)
    #    test_cloud.generate_statistical_removal_and_average_distance_data(100, 10, 3, 10, 1, .00001, graph=True)
    #    test_cloud.clean_noisy_cloud(method='Average distance', show=True)

#     test_cloud_stl = pcc.PointCloudStructure(stl_path='test.stl', stl_sample_points=1000)
#     test_cloud_stl.generate_gaussian_noise(0, 3, .20, True)
#     test_cloud_stl.transform_noisy_cloud(show=True)
#     test_cloud_stl.generate_statistical_removal_and_average_distance_data(100, 10, 3, 8, 1, .00001)
#     test_cloud_stl.clean_noisy_cloud(method='Average distance')
#     test_cloud_stl.align_cleaned_cloud_and_original_cloud(show=True, voxel_size=2)