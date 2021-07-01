"""
In this file we want to implement a numpythonic version of the previous two
programs. Here we wish to vectorize loops that take a long time, mainly the
loops that go over the number of samples as this grows very large. Using the
cprofilev tool we can see which functions / loops are the biggest bottlenecks in
our program and subsequently work to try to vectorize these.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import *



def get_distances_to_clusters(data, centroids):
    """
    Squared Euclidean distance between all data-points and every centroid. For
    the function to work properly it needs data and centroids to be numpy
    broadcastable. We sum along the dimension axis.

    Inputs:
        data (np.array): with dimensions (samples x 1 x dim)
        centroids (np.array): with dimensions (1 x n_clusters x dim)

    Returns:
        distances (np.array): with dimensions (samples x n_clusters)
    """
    distances = np.sum(np.abs((data - centroids))**2, axis=2)
    return distances



def assign_points_to_clusters(distances):
    """
    Assigning each data-point to a cluster given an array distances containing
    the squared Euclidean distance from every point to each centroid. We do
    np.argmin along the cluster axis to find the closest cluster. Returns a
    numpy array with corresponding labels.

    Inputs:
        distances (np.array): with dimensions (samples x n_clusters)

    Returns:
        cluster_labels (np.array): with dimensions (samples x None)
    """
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels



def k_means(data, n_clusters=4, max_iterations=100, tolerance=1e-8,
            progression_plot=True):
    """
    Numpythonic implementation of the k-means clusting algorithm.

    Inputs:
        data (np.array): with dimesions (samples x dim)
        n_clusters (int): hyperparameter which depends on dataset
        max_iterations (int): hyperparameter which depends on dataset
        tolerance (float): convergence measure
        progression_plot (bool): activation flag for plotting

    Returns:
        cluster_labels (np.array): with dimension (samples)
        centroid_list (list): list of centroids (np.array)
                              with dimensions (n_clusters x dim)
    """
    start_time = time.time()

    n_samples, dimensions = data.shape
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]

    distances = get_distances_to_clusters(np.reshape(data,
                                            (n_samples, 1, dimensions)),
                                          np.reshape(centroids,
                                            (1, n_clusters, dimensions)))
    cluster_labels = assign_points_to_clusters(distances)

    #centroids_list = []
    #temp_centroids = centroids.copy()
    #centroids_list.append(temp_centroids)


    for iteration in range(max_iterations):
        prev_centroids = centroids.copy()
        for k in range(n_clusters):
            points_in_cluster = data[cluster_labels == k]
            mean_vector = np.mean(points_in_cluster, axis=0)
            centroids[k] = mean_vector

        #temp_centroids = centroids.copy()
        #centroids_list.append(temp_centroids)
        distances = get_distances_to_clusters(np.reshape(data,
                                                (n_samples, 1, dimensions)),
                                              np.reshape(centroids,
                                                (1, n_clusters, dimensions)))
        cluster_labels = assign_points_to_clusters(distances)


        if progression_plot:
            unique_cluster_labels = np.unique(cluster_labels)
            #fig, ax = plt.subplots(figsize=(4, 4))

            fig = plt.figure(figsize=(4, 4))
            if dimensions == 3:
                ax = fig.add_subplot(projection='3d')

            elif dimensions == 2:
                ax = fig.add_subplot()

            else:
                print('cant plot in this other dimensions')

            file_folder = './clustering_example_images/'
            for i in unique_cluster_labels:
                ax.scatter(data[cluster_labels == i, 0],
                        data[cluster_labels == i, 1],
                        label = i,
                        alpha = 0.2)

                ax.scatter(centroids[:, 0], centroids[:, 1], c='black')

                ax.set_title(f'Clusters at iteration {iteration}')
                #fig.tight_layout()
                #fig.legend()
                plt.savefig(file_folder +
                            f'c_image_at_it_{str(iteration).zfill(3)}.png')
            plt.close()


        centroid_difference = np.sum(np.abs(centroids - prev_centroids))
        if centroid_difference < tolerance:
            print(f'Converged at iteration: {iteration}')
            print(f'Runtime: {time.time() - start_time} seconds')

            return cluster_labels, centroids


    print(f'Did not converge in {max_iterations} iterations')
    print(f'Runtime: {time.time() - start_time} seconds')

    return cluster_labels, centroids_list


if __name__=='__main__':
    np.random.seed(2021)
    simple_data = generate_simple_clustering_dataset(n_points=1000, plotting=False)
    complicated_data = generate_complicated_clustering_dataset(n_points=1000, plotting=False)
    k_means(simple_data, n_clusters=4, progression_plot=False)
