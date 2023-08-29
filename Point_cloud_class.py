import open3d as o3d
import numpy as np
from sklearn import decomposition as decomp
from matplotlib import pyplot as plt
from matplotlib import cm
import random
import copy

class PointCloud:
    """Datastructure to hold point cloud data and perform operations on it.
    """
    def __init__(self, pointCloud: o3d.geometry.PointCloud = 0, pointCloudFilePath: str = 0):
        """Datastructure to hold point cloud data and perform operations on it.

        Args:
            pointCloud (o3d.geometry.PointCloud, optional): Open3d point cloud to create this
            class from. Defaults to 0. \n
            pointCloudFilePath (str, optional): File path to use to create this class. Either give
            this argument or a pointCloud. Defaults to 0.
        """
        if pointCloud != 0:
            self.PointCloud = pointCloud
        elif pointCloudFilePath != 0:
            self.PointCloud = o3d.io.read_point_cloud(pointCloudFilePath)
        self.noisy_cloud = None
        
        print("Showing point cloud")
        o3d.visualization.draw_geometries([self.PointCloud])
        
    def __str__(self) -> str:
        """Provides summary of the point cloud.

        Returns:
            str: String with info about the point cloud.
        """
        return str(self.PointCloud)
    
    def generate_statistical_removal_and_PCA_variance_data(self, nb_neighbor_upper: int, nb_neighbors_samples: int, std_dev_upper: float,
                                               std_dev_samples: int, nb_neighbor_lower: int = 1,
                                               std_dev_lower: float = .000001, graph: bool = False) -> None:
        """Graphs the variance of the point cloud along PCA components as a function of statistical outlier removal.
        Stores the parameters that give the least variance ratio.

        Args:
            nb_neighbor_upper (int): Upper bound for number of neighbors argument\n
            nb_neighbors_samples (int): Number of samples to test for nb_neighbors\n
            std_dev_upper (float): upper bound for standard deviation argument\n
            std_dev_samples (int): Number of samples to use for standard deviation\n
            nb_neighbor_lower (int, optional): lower bound for number of neighbors argument. Default 1.\n
            std_dev_lower (float, optional): Lower bound for standard deviation argument. Default .000001\n
            graph (bool, optional): Whether or not to graph the data. Default False.

        Raises:
            Exception: If you try to run this function without generating a noisy cloud first, it will raise an exception.
        """
        
        if self.noisy_cloud == None:
            raise Exception("No noisy cloud has been generated. Please generate a noisy cloud before calling\
                            this function, by running PointCloud.generate_gaussian_noise.")
        
        self.best_least_variance_ratio = np.inf
        
        pcd = self.noisy_cloud
        self.nb_neighbor_points = np.linspace(nb_neighbor_lower, nb_neighbor_upper, nb_neighbors_samples).astype(int)
        self.std_dev_points = np.linspace(std_dev_lower, std_dev_upper, std_dev_samples)
        self.explained_variance_ratio_data = np.empty((3, self.nb_neighbor_points.size, self.std_dev_points.size))
        self.explained_variance_data = np.empty((3, self.nb_neighbor_points.size, self.std_dev_points.size))
        for i in range(self.nb_neighbor_points.size):
            for j in range(self.std_dev_points.size):
                downpcd, _ = pcd.remove_statistical_outlier(nb_neighbors=self.nb_neighbor_points[i],
                                                            std_ratio=self.std_dev_points[j])
                points = np.asarray(downpcd.points)
                if points.size != 0:
                    PCA = decomp.PCA(n_components=3)
                    PCA.fit(points)
                    for k in range(3):
                        self.explained_variance_ratio_data[k, i, j] = PCA.explained_variance_ratio_[k]
                        self.explained_variance_data[k, i, j] = PCA.explained_variance_[k]
                        if PCA.explained_variance_ratio_[k] < self.best_least_variance_ratio:
                            self.best_least_variance_ratio = PCA.explained_variance_ratio_[k]
                            self.best_nb_neighbors_PCA = self.nb_neighbor_points[i]
                            self.best_std_dev_PCA = self.std_dev_points[j]
                else:
                    for k in range(3):
                        self.explained_variance_data[k, i, j] = 'nan'
                        self.explained_variance_ratio_data[k, i, j] = 'nan'
        self.nb_neighbor_points, self.std_dev_points = np.meshgrid(self.nb_neighbor_points, self.std_dev_points)
        graph_labels_ratio = ['Principal Component Highest Variance Ratio', 'Principal Component Medium Variance Ratio',
                            'Principal Component Least Variance Ratio']
        graph_labels_amount = ['Variance Highest', 'Variance Medium', 'Variance Lowest']
        
        if graph == True:
            for i in range(3):
                fig = plt.figure(figsize=(8, 4))
                ax = fig.add_subplot(1, 2, 1, projection='3d')
                ax.plot_surface(self.std_dev_points, self.nb_neighbor_points, np.transpose(self.explained_variance_ratio_data[i, :, :]),
                                    cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False)
                ax.set_title(graph_labels_ratio[i])
                ax.set_xlabel('Standard deviation ratio')
                ax.set_ylabel('Number of neighbors')
                ax.set_zlabel('Variance ratio')
                ax.view_init(azim=160, elev=30)
                ax = fig.add_subplot(1, 2, 2, projection='3d')
                ax.plot_surface(self.std_dev_points, self.nb_neighbor_points, np.transpose(self.explained_variance_data[i, :, :]),
                                    cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False)
                ax.set_title(graph_labels_amount[i])
                ax.set_xlabel('self.std_dev_points')
                ax.set_ylabel('# of neighbors')
                ax.set_zlabel('Variance Ratio')
                ax.view_init(azim=160, elev=30)
            plt.show()
        return

    def generate_statistical_removal_and_average_distance_data(self, nb_neighbor_upper: int, nb_neighbors_samples: int, std_dev_upper: float,
                                                   std_dev_samples: int, nb_neighbor_lower: int = 1,
                                                   std_dev_lower: float = .000001, graph: bool = False) -> None:
        """Graphs the average distance between points as a function of statistical outlier removal. 
        Stores the parameters that give the least average distance.

        Args:
            nb_neighbor_upper (int): Upper bound for number of neighbors argument\n
            nb_neighbors_samples (int): Number of samples to test for nb_neighbors\n
            std_dev_upper (float): Upper bound for standard deviation argument\n
            std_dev_samples (int): Number of samples to use for standard deviation\n
            nb_neighbor_lower (int, optional): Lower bound for number of neighbors argument. Default 1.\n
            std_dev_lower (float, optional): Lower bound for standard deviation argument. Default .000001\n
            graph (bool, optional): Whether or not to graph the data. Default False.

        Raises:
            Exception: If you try to run this function without generating a noisy cloud first, it will raise an exception.  
        """
        
        if self.noisy_cloud == None:
            raise Exception("No noisy cloud has been generated. Please generate a noisy cloud before calling\
                        this function, by running PointCloud.generate_gaussian_noise.")
        
        self.best_mean_distance = np.inf
        
        self.nb_neighbor_points_ad = np.linspace(nb_neighbor_lower, nb_neighbor_upper, nb_neighbors_samples).astype(int)
        self.std_dev_points_ad = np.linspace(std_dev_lower, std_dev_upper, std_dev_samples)
        self.average_distance_data = np.empty((self.nb_neighbor_points_ad.size, self.std_dev_points_ad.size))
        for i in range(self.nb_neighbor_points_ad.size):
            for j in range(self.std_dev_points_ad.size):
                downpcd, _ = self.noisy_cloud.remove_statistical_outlier(nb_neighbors=self.nb_neighbor_points_ad[i], std_ratio=self.std_dev_points_ad[j])
                points = np.asarray(downpcd.points)
                if points.size != 0:
                    mean_distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(downpcd))
                    if mean_distance < self.best_mean_distance:
                        self.best_mean_distance = mean_distance
                        self.best_nb_neighbors_mean = self.nb_neighbor_points_ad[i]
                        self.best_std_dev_mean = self.std_dev_points_ad[j]
                    self.average_distance_data[i, j] = mean_distance
                else:
                    self.average_distance_data[i, j] = 'nan'
        self.nb_neighbor_points_ad, self.std_dev_points_ad = np.meshgrid(self.nb_neighbor_points_ad, self.std_dev_points_ad)
        
        if graph == True:
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.nb_neighbor_points_ad, self.std_dev_points_ad, np.transpose(self.average_distance_data),
                                cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_title('Average Distance Between Points')
            ax.set_xlabel('Number of neighbors')
            ax.set_ylabel('Standard deviation')
            ax.set_zlabel('Average distance')
            plt.show()
        return
    
    def generate_gaussian_noise(self, mean: float, std_dev_distance_mult: float, noise_percent: float) -> None:
        """Generates Gaussian distributed noise for a point cloud.

        Args:
            mean (float):  Mean distance the noise moves from the original point.\n
            std_dev_distance_mult (float): Standard deviation of the original points to the noise generated. Measured by taking
            this argument and multiplying it by the average distance between points in the point cloud.\n
            noise_percent (float): What percentage of points are moved from their original location.
        """    
            
        def display_noise_not_noise(cloud, ind):
            noise_cloud = cloud.select_by_index(ind)
            correct_cloud = cloud.select_by_index(ind, invert=True)
            noise_cloud.paint_uniform_color([1, 0, 0])
            correct_cloud.paint_uniform_color([0.8, 0.8, 0.8])
            print('Noise points are shown in red, correct points are shown in gray.')
            o3d.visualization.draw_geometries([noise_cloud, correct_cloud])
        
        self.noise_indeces = []
        points = np.array([1, 1])
        noise_counter = 0
        pcd = self.PointCloud
        points = copy.copy(np.asarray(self.PointCloud.points))
        distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(pcd))
        std_dev = distance*std_dev_distance_mult
        for i in range(len(points)):
            if random.random() <= noise_percent:
                self.noise_indeces.append(i)
                noise_counter += 1
                noise = np.random.normal(mean, std_dev, (1, 3))
                points[i] = points[i] + noise
        self.noisy_cloud = o3d.geometry.PointCloud()
        self.noisy_cloud.points = o3d.utility.Vector3dVector(points)
        self.noise_counter = noise_counter
        display_noise_not_noise(self.noisy_cloud, self.noise_indeces)
        
    def evaluate_filter_parameters(self, nb_neighbors = 0, std_ratio = 0, method:str = None, show_images:bool = False) -> (float):
        """Evaluate the filter quality for the statistical outlier removal filter. Reports accuracy, recall, precision, 
        and Normalized RMSE = RMSE/(average distance between points in the original cloud). Finally, reports a 
        summary statistic which is average(precision, recall, accuracy)*(average distance between points in the original cloud)/(RMSE)
        Note: MVP point clouds always start with 2048 points.

        Args:
            nb_neighbors (int, optional): nb_neighbors argument to use for the statistical outlier removal filter. Defaults to 0.\n
            std_ratio (int, optional): std_ratio argument to use for the statistical outlier removal filter. Defaults to 0.\n
            method (str, optional): Use either 'PCA' or 'Average distance'. This will automatically select the hypothetically 'best'
            filter parameters using the selected method. Defaults to None.
            show_images (bool, optional): Whether or not to show the original, noisy, and filtered point clouds. Defaults to False.
            
        Returns:
            float: Summary statistic for the filter quality. This is defined as average(precision, recall, accuracy)*(average distance between points in the original cloud)/(RMSE)
            
        Raises:
            Exception: If you run this function without providing either nb_neighbors and std_ratio or method, it will raise an exception.
        """ 
               
        if (nb_neighbors == 0 or std_ratio == 0) and method == None:
            raise Exception('Please provide either nb_neighbors and std_ratio or method for filter evaluation.')
        
        if method == 'PCA':
            nb_neighbors = self.best_nb_neighbors_PCA
            std_ratio = self.best_std_dev_PCA
            
        elif method == 'Average distance':
            nb_neighbors = self.best_nb_neighbors_mean
            std_ratio = self.best_std_dev_mean
        
        pcd_in = copy.copy(self.PointCloud)
        if show_images == True:
            print("Showing original point cloud")
            o3d.visualization.draw_geometries([pcd_in])
        pcd_noisy = self.noisy_cloud
        if show_images == True:
            print("Showing noisy point cloud")
            o3d.visualization.draw_geometries([pcd_noisy])
        clean_array = np.asarray(pcd_in.points)
        noisy_array = np.asarray(pcd_noisy.points)
        noise_point_original = 0
        for point in range(np.size(clean_array, 0)):
            if (noisy_array[point] - clean_array[point] != [0, 0, 0]).any():
                noise_point_original += 1
        self.cleaned_cloud, ind = pcd_noisy.remove_statistical_outlier(nb_neighbors, std_ratio)
        if show_images == True:
            print("Showing filtered point cloud")
            o3d.visualization.draw_geometries([self.cleaned_cloud])
        noise_point_new = 0
        for point in range(np.size(clean_array, 0)):
            if ind.count(point) == 1:
                if (noisy_array[point] - clean_array[point] != [0, 0, 0]).any():
                    noise_point_new += 1

        original_cloud_size = len(self.PointCloud.points)
        noise_points_removed = noise_point_original - noise_point_new
        total_points_removed = original_cloud_size - len(ind)
        noise_detection_rate = noise_points_removed / noise_point_original * 100
        correct_points_removed = total_points_removed - noise_points_removed
        correct_points_left = original_cloud_size - noise_point_original - (total_points_removed - noise_points_removed)
        average_point_distance = np.mean(o3d.geometry.PointCloud.compute_nearest_neighbor_distance(self.PointCloud))
        recall = noise_points_removed / (noise_points_removed + correct_points_removed) * 100
        accuracy = (noise_points_removed + correct_points_left) / (2048) * 100
        RMSE = np.sqrt(np.sum((clean_array[ind] - self.cleaned_cloud.points) ** 2) / np.size(clean_array, 0))/average_point_distance

        if show_images == True:
            print('recall: ' + str(recall) + '%')
            print('accuracy: ' + str(accuracy) + '%')
            print('noise detection rate: ' + str(noise_detection_rate) + '%')
            print('Normalized RMSE: ' + str(RMSE))
        
        if RMSE != 0:
            if show_images == True:
                print('Summary statistic: ' + str(np.average([recall, np.float64(accuracy), noise_detection_rate]) / (RMSE)))
            return np.average([recall, np.float64(accuracy), noise_detection_rate]) / (RMSE)
        
        else:
            print('RMSE was zero for nb_neighbors = ' + str(nb_neighbors) + ' and std_ratio = ' + str(std_ratio) + '.')
            return np.nan
        
    def generate_summary_statistic_data(self, nb_neighbor_upper: int, nb_neighbors_samples: int, std_dev_upper: float,
                                               std_dev_samples: int, nb_neighbor_lower: int = 1,
                                               std_dev_lower: float = .000001, graph: bool = False) -> None:
        """Generate data for the summary statistic over the input domain. The summary statistic is defined as
        average(precision, recall, accuracy)*(average distance between points in the original cloud)/(RMSE).

        Args:
            nb_neighbor_upper (int): Upper bound for number of neighbors argument\n
            nb_neighbors_samples (int): Number of samples to test for nb_neighbors\n
            std_dev_upper (float): Upper bound for standard deviation argument\n
            std_dev_samples (int): Number of samples to use for standard deviation\n
            nb_neighbor_lower (int, optional): Lower bound for number of neighbors argument. Default 1.\n
            std_dev_lower (float, optional): Lower bound for standard deviation argument. Default .000001\n
            graph (bool, optional): Whether or not to graph the data. Default False.
        """        
        if self.noisy_cloud == None:
            raise Exception("No noisy cloud has been generated. Please generate a noisy cloud before calling\
                        this function, by running PointCloud.generate_gaussian_noise.")
        
        self.best_summary_statistic = 0
        
        self.nb_neighbor_points_ss = np.linspace(nb_neighbor_lower, nb_neighbor_upper, nb_neighbors_samples).astype(int)
        self.std_dev_points_ss = np.linspace(std_dev_lower, std_dev_upper, std_dev_samples)
        self.summary_statistic_data = np.empty((self.nb_neighbor_points_ss.size, self.std_dev_points_ss.size))
        for i in range(self.nb_neighbor_points_ss.size):
            for j in range(self.std_dev_points_ss.size):
                self.summary_statistic_data[i, j] = self.evaluate_filter_parameters(nb_neighbors=self.nb_neighbor_points_ss[i], std_ratio=self.std_dev_points_ss[j])
                if self.summary_statistic_data[i, j] > self.best_summary_statistic:
                    self.best_summary_statistic = self.summary_statistic_data[i, j]
                    self.best_nb_neighbors_ss = self.nb_neighbor_points_ss[i]
                    self.best_std_dev_ss = self.std_dev_points_ss[j]
        self.nb_neighbor_points_ss, self.std_dev_points_ss = np.meshgrid(self.nb_neighbor_points_ss, self.std_dev_points_ss)
        
        self.optimal_filtered_cloud, _ = self.noisy_cloud.remove_statistical_outlier(self.best_nb_neighbors_ss, self.best_std_dev_ss)
        
        if graph == True:
            print('Graphing summary statistic over input domain')
            ax = plt.axes(projection='3d')
            ax.plot_surface(self.nb_neighbor_points_ss, self.std_dev_points_ss, np.transpose(self.summary_statistic_data),
                                cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_title('Summary statistic over input domain')
            ax.set_xlabel('Number of neighbors')
            ax.set_ylabel('Standard deviation')
            ax.set_zlabel('Summary statistic')
            plt.show()
        return

    
    def create_summary_subplots(self, method: str):
        """Creates a matplotlib figure with 5 subplots. The first subplot is the original point cloud. The second subplot is the noisy point cloud.
        The third subplot is a graph representing output from either the average distance method or 
        the PCA method, depending on which method is selected. The fourth subplot is a graph
        representing the summary statistic. The fifth subplot is the filtered point cloud.

        Args:
            method (str): Method for graphing the third subplot. Either 'PCA' or 'Average distance'.

        Raises:
            Exception: If you try to run this function without selecting a method, it will raise an exception.
        """        
        if method != 'PCA' and method != 'Average distance':
            raise Exception('method must be either "PCA" or "Average distance"')
        
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(2,3,1, projection='3d')
        
        #Graphing original point cloud
        
        (ax).set_title('Original Point Cloud')
        point_cloud_points = np.asarray(self.PointCloud.points)
        (ax).scatter(point_cloud_points[:,0], point_cloud_points[:,1], point_cloud_points[:,2], s=.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.view_init(azim=20, elev=20)
        
        #Graphing noisy point cloud
        
        ax = fig.add_subplot(2,3,2, projection='3d')
        (ax).set_title('Noisy Point Cloud')
        point_cloud_points = np.asarray(self.noisy_cloud.points)
        (ax).scatter(point_cloud_points[self.noise_indeces,0], point_cloud_points[self.noise_indeces,1], point_cloud_points[self.noise_indeces,2], s=.5, c='r')
        noise_mask = np.ones(len(point_cloud_points), dtype=bool)
        noise_mask[self.noise_indeces] = False
        non_noise = point_cloud_points[noise_mask]
        (ax).scatter(non_noise[: ,0], non_noise[: ,1], non_noise[: ,2], s=.5, c=[[.7,.7,.7]])
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.view_init(azim=20, elev=20)
        
        #Graphing minimum of PCA method of filtering
        
        if method == 'PCA':
        
            ax = fig.add_subplot(2,3,3, projection='3d')
            ax.set_title('Principal Component\nLeast Variance Ratio')
            ax.plot_surface(self.std_dev_points, self.nb_neighbor_points, np.transpose(self.explained_variance_ratio_data[2, :, :]),
                                    cmap=cm.coolwarm,
                                    linewidth=0, antialiased=False)
            ax.set_xlabel('Standard deviation ratio')
            ax.set_ylabel('Number of neighbors')
            ax.set_zlabel('Variance ratio')
            ax.view_init(azim=160, elev=30)
            ax.yaxis.labelpad=5
            ax.xaxis.labelpad=5
            ax.zaxis.labelpad=8
            
        #Graphing minimum of average distance method of filtering    
            
        elif method == 'Average distance':
        
            ax = fig.add_subplot(2,3,3, projection='3d')
            ax.plot_surface(self.nb_neighbor_points_ad, self.std_dev_points_ad, np.transpose(self.average_distance_data),
                                cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
            ax.set_title('Average Distance\n Between Points')
            ax.set_xlabel('Number of neighbors')
            ax.set_ylabel('Standard deviation')
            ax.set_zlabel('Average distance')
            ax.view_init(azim=160, elev=30)
            ax.yaxis.labelpad=5
            ax.xaxis.labelpad=5
            ax.zaxis.labelpad=8
            
        #Graphing summary statistic over input domain
            
        ax = fig.add_subplot(2,3,4, projection='3d')
        ax.set_title('Summary Statistic\n over Input Domain')
        ax.plot_surface(self.nb_neighbor_points_ss, self.std_dev_points_ss, np.transpose(self.summary_statistic_data),
                            cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        ax.set_title('Summary statistic over input domain')
        ax.set_xlabel('Number of neighbors')
        ax.set_ylabel('Standard deviation')
        ax.set_zlabel('Summary statistic')
        ax.view_init(azim=60, elev=20)    
            
        #Graphing filtered point cloud using the selected method
            
        ax = fig.add_subplot(2,3,5, projection='3d')
        ax.set_title('Filtered Point Cloud\n with ' + method + ' Method')
        cleaned_point_cloud_points = np.asarray(self.cleaned_cloud.points)
        ax.scatter(cleaned_point_cloud_points[:,0], cleaned_point_cloud_points[:,1], cleaned_point_cloud_points[:,2], s=.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.view_init(azim=20, elev=20)
        
        #Graphing filtered point cloud using the optimal parameters for the summary statistic
        ax = fig.add_subplot(2,3,6, projection='3d')
        ax.set_title('Best Filtered Point Cloud\n by the Summary Statistic')
        cleaned_point_cloud_points = np.asarray(self.optimal_filtered_cloud.points)
        ax.scatter(cleaned_point_cloud_points[:,0], cleaned_point_cloud_points[:,1], cleaned_point_cloud_points[:,2], s=.5)
        ax.set_aspect('equal', adjustable='box')
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_zticklabels([])
        ax.view_init(azim=20, elev=20)
        
        #plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.show()
            