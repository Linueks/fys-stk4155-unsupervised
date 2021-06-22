"""
Program naively implementing k-means algorithm from scratch in Python
most basic implementation but in this file we have defined some functions for
readability.
--------------------------------------------------------------------------------
Written Summer 2021 Linus Ekstrom for FYS-STK4155 course content. A lot like the
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import *
#plt.style.use('ggplot')





def get_distances_to_clusters(data, centroids):
    """
    Function that for each cluster finds the squared Euclidean distance
    from every data point to the cluster centroid and returns a numpy array
    containing the distances such that distance[i, j] means the distance between
    the i-th point and the j-th centroid.

    Inputs:
        data (np.array): with dimensions (samples x dim)
        centroids (np.array): with dimensions (n_clusters x dim)

    Returns:
        distances (np.array): with dimensions (samples x n_clusters)
    """

    samples, dimensions = data.shape
    n_clusters = centroids.shape[0]
    distances = np.zeros((samples, n_clusters))
    for k in range(n_clusters):
        for i in range(samples):
            dist = 0
            for j in range(dimensions):
                dist += np.abs(data[i, j] - centroids[k, j])**2
                distances[i, k] = dist

    return distances



def assign_points_to_clusters(distances):
    """
    Function to assign each data point to the cluster to which it is the closest
    based on the squared Euclidean distance from the get_distances_to_clusters
    method.

    Inputs:
        distances (np.array): with dimensions (samples x n_clusters)

    Returns:
        cluster_labels (np.array): with dimensions (samples)
    """
    samples, n_clusters = distances.shape
    cluster_labels = np.zeros(samples, dtype='int')
    # basically just a self written argmin
    for i in range(samples):
        # track keeping variable for finding the smallest value in each column
        # set to big number to avoid missing numbers
        smallest = 1e10
        smallest_row_index = 1e10
        for k in range(n_clusters):
            #print(distances[k, i])
            if distances[i, k] < smallest:
                smallest = distances[i, k]
                smallest_row_index = k

        cluster_labels[i] = smallest_row_index

    return cluster_labels



def k_means(data, n_clusters=4, max_iterations=20, tolerance=1e-8,
            plot_results=False):
    """
    Naive implementation of the k-means clustering algorithm. A short summary of
    the algorithm is as follows: we randomly initialize k centroids / means.
    Then we assign, using the squared Euclidean distance, every data-point to a
    cluster. We then update the position of the k centroids / means, and repeat
    until convergence or we reach our desired maximum iterations. The method
    returns the cluster assignments of our data-points and a sequence of
    centroids.

    Inputs:
        data (np.array): with dimesions (samples x dim)
        n_clusters (int): hyperparameter which depends on dataset
        max_iterations (int): hyperparameter which depends on dataset
        tolerance (float): convergence measure
        plot_results (bool): activation flag for plotting

    Returns:
        cluster_labels (np.array): with dimension (samples)
        centroid_list (list): list of centroids (np.array)
                              with dimensions (n_clusters x dim)
    """
    samples, dimensions = data.shape
    # we need to (randomly) choose initial centroids
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]
    distances = get_distances_to_clusters(data, centroids)
    cluster_labels = assign_points_to_clusters(distances)

    if plot_results:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].scatter(data[:, 0], data[:, 1])
        axs[0, 0].set_title("Toy Model Dataset")


        axs[0, 1].scatter(data[:, 0], data[:, 1])
        axs[0, 1].scatter(centroids[:, 0], centroids[:, 1])
        axs[0, 1].set_title("Initial Random Centroids")


        unique_cluster_labels = np.unique(cluster_labels)
        for i in unique_cluster_labels:
            axs[1, 0].scatter(data[cluster_labels == i, 0],
                                data[cluster_labels == i, 1],
                                label = i)

        axs[1, 0].set_title("First Grouping of Points to Centroids")

    centroid_list = []
    temp_centroids = centroids.copy()
    centroid_list.append(temp_centroids)
    # For each cluster we need to update the centroid by calculating new means
    # for all the data points in the cluster and repeat
    for iteration in range(max_iterations):
        for k in range(n_clusters):
            # Here we find the mean for each centroid
            vector_mean = np.zeros(dimensions)
            n = 0
            for i in range(samples):
                if cluster_labels[i] == k:
                    vector_mean += data[i, :]
                    n += 1
            # And update according to the new means
            centroids[k, :] = vector_mean / n
            distances = np.zeros((samples, n_clusters))

        # we need to use copies to avoid overwriting (pointer stuff)
        temp_centroids = centroids.copy()
        centroid_list.append(temp_centroids)

        # we find the squared Euclidean distance from each centroid to every point
        distances = get_distances_to_clusters(data, centroids)
        # we assign each point to a cluster
        cluster_labels = assign_points_to_clusters(distances)

        centroid_difference = np.sum(np.abs(centroid_list[iteration] - centroid_list[iteration-1]))
        if centroid_difference < tolerance:
            print(f'Converged at iteration: {iteration}')
            if plot_results:
                unique_cluster_labels = np.unique(cluster_labels)
                for i in unique_cluster_labels:
                    axs[1, 1].scatter(data[cluster_labels == i, 0],
                                data[cluster_labels == i, 1],
                                label = i)

                axs[1, 1].set_title("Final Grouping")
                fig.tight_layout()
                plt.show()

            return cluster_labels, centroid_list

    if plot_results:
        unique_cluster_labels = np.unique(cluster_labels)
        for i in unique_cluster_labels:
            axs[1, 1].scatter(data[cluster_labels == i, 0],
                        data[cluster_labels == i, 1],
                        label = i)

        axs[1, 1].set_title("Final Grouping")
        fig.tight_layout()
        plt.show()

    print(f'Did not converge in {max_iterations} iterations')
    return cluster_labels, centroid_list



if __name__=='__main__':
    np.random.seed(2021)
    data = generate_clustering_dataset(dim=2, n_points=1000, plotting=False, return_data=True)
    cluster_labels, centroids = k_means(data, max_iterations=20, plot_results=True)
