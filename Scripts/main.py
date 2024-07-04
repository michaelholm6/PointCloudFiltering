import Point_cloud_class as pcc
import cProfile
import pstats
import winsound
import random
import time

def main(clouds_to_select, runs_per_cloud, input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper, nb_neighbors_samples, std_dev_upper, std_dev_samples, nb_neighbors_lower, std_dev_lower, folder_path):
        pointCloudCollection = pcc.PointCloudDatabase(input_folder_path, noise_mean, std_dev_distance_mult, noise_percent, nb_neighbors_upper,
                                                      nb_neighbors_samples, std_dev_upper, std_dev_samples,
                                                          nb_neighbors_lower, std_dev_lower)
        pointCloudCollection.record_point_cloud_information(folder_path=folder_path, runs_per_cloud=runs_per_cloud, clouds_to_select=clouds_to_select)

if __name__ == "__main__":
    
    print("Starting .05.20 at ", time.strftime('%H:%M:%S', time.localtime(time.time())))
    main(50, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .20, 100, 10, 3, 10, 1, .00001, '.05.20new2')
    print("Starting .10.20 at ", time.strftime('%H:%M:%S', time.localtime(time.time())))
    main(50, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .10, .20, 100, 10, 3, 10, 1, .00001, '.10.20new')
    print("Starting .05.10 at ", time.strftime('%H:%M:%S', time.localtime(time.time())))
    main(50, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .10, 100, 10, 3, 10, 1, .00001, '.05.10new')
    print("Starting .05.30 at ", time.strftime('%H:%M:%S', time.localtime(time.time())))
    main(50, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .05, .30, 100, 10, 3, 10, 1, .00001, '.05.30new')
    print("Starting .01.20 at ", time.strftime('%H:%M:%S', time.localtime(time.time())))
    main(50, 10, r'C:\Users\Michael\Desktop\VRAC\Boeing Digital Twin\Journal Article\Code\PointCloudFiltering\3D_files\MVP Point Clouds', 0, .01, .20, 100, 10, 3, 10, 1, .00001, '.01.20new')