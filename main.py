import Point_cloud_class as pcc

test_cloud = pcc.PointCloud(pointCloudFilePath="3D_files/Point Clouds/pcd10.pcd")
test_cloud.generate_gaussian_noise(mean=0, std_dev_distance_mult=3, noise_percent=0.2)
test_cloud.graph_statistical_removal_and_PCA_variance(50, 10, 3, 5, 1, .00001)
test_cloud.graph_statistical_removal_and_average_distance(50, 10, 3, 5, 1, .00001)
test_cloud.evaluate_filter_parameters(method='Average distance')
test_cloud.create_summary_subplots(method='Average distance')