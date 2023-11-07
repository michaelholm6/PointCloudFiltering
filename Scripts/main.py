import Point_cloud_class as pcc
import cProfile
import pstats

def main():
        pointCloudCollection = pcc.PointCloudDatabase('3D_files/MVP Point Clouds')
        pointCloudCollection.record_point_cloud_information(folder_path='test')

if __name__ == "__main__":
    
    main()

    # cProfile.run('main()', 'my_func_stats')
    
    # stream = open('output.txt', 'w')
    # p = pstats.Stats("my_func_stats", stream=stream)
    # p.sort_stats("cumulative").print_stats()
        
    
    
    # test_cloud = pcc.PointCloudStructure(pointCloudFilePath=r"C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds\pcd1200.ply")
    # test_cloud.generate_gaussian_noise(0, 3, .20, True)
    # test_cloud.transform_noisy_cloud(show=True)
    # test_cloud.generate_statistical_removal_and_average_distance_data(100, 10, 3, 8, 1, .00001)
    # test_cloud.clean_noisy_cloud(method='Average distance')
    # test_cloud.align_cleaned_cloud_and_original_cloud(show=True)

    # test_cloud_stl = pcc.PointCloudStructure(stl_path='test.stl', stl_sample_points=1000)
    # test_cloud_stl.generate_gaussian_noise(0, 3, .20, True)
    # test_cloud_stl.transform_noisy_cloud(show=True)
    # test_cloud_stl.generate_statistical_removal_and_average_distance_data(100, 10, 3, 8, 1, .00001)
    # test_cloud_stl.clean_noisy_cloud(method='Average distance')
    # test_cloud_stl.align_cleaned_cloud_and_original_cloud(show=True, voxel_size=2)