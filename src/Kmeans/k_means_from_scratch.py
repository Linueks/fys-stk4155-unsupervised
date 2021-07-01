"""
Program naively implementing k-means algorithm from scratch in Python
This file is the absolute 'simplest' implementation I could do of k-means.
However, I feel some clarity is lost when not using functions whether custom
or from various libraries.
--------------------------------------------------------------------------------
Written Summer 2021 Linus Ekstrom for FYS-STK4155 course content
"""
import numpy as np
import matplotlib.pyplot as plt
from generate_dataset import *
#plt.style.use('ggplot')



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

    Returns:
        cluster_labels (np.array): with dimension (samples)
        centroid_list (list): list of centroids (np.array)
                              with dimensions (n_clusters x dim)
    """
    start_time = time.time()

    samples, dimensions = data.shape
    # we need to (randomly) choose initial centroids
    print(samples, len(data))
    centroids = data[np.random.choice(samples, n_clusters, replace=False), :]
    # next we initialize an array to hold the distance from each centroid to
    # every point in our dataset
    distances = np.zeros((samples, n_clusters))

    # we find the squared Euclidean distance from each centroid to every point
    for k in range(n_clusters):
        for i in range(samples):
            dist = 0
            for j in range(dimensions):
                dist += np.abs(data[i, j] - centroids[k, j])**2
                distances[i, k] = dist

    # then we need to assign each data point to their closest centroid
    # We initialize a list to put our points into clusters. This way we do it here
    # the index signifies which point and the value which cluster
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


    #centroid_list = []
    #temp_centroids = centroids.copy()
    #centroid_list.append(temp_centroids)
    # For each cluster we need to update the centroid by calculating new means
    # for all the data points in the cluster and repeat
    for iteration in range(max_iterations):
        prev_centroids = centroids.copy()
        for k in range(n_clusters):
            # Here we find the mean for each centroid
            vector_mean = np.zeros(dimensions)
            mean_divisor = 0
            for i in range(samples):
                if cluster_labels[i] == k:
                    vector_mean += data[i, :]
                    mean_divisor += 1
            # And update according to the new means
            centroids[k, :] = vector_mean / mean_divisor
            #distances = np.zeros((samples, n_clusters))

        # we need to use copies to avoid overwriting (pointer stuff)
        #temp_centroids = centroids.copy()
        #centroid_list.append(temp_centroids)

        # we find the squared Euclidean distance from each centroid to every point
        for k in range(n_clusters):
            for i in range(samples):
                dist = 0
                for j in range(dimensions):
                    dist += np.abs(data[i, j] - centroids[k, j])**2
                    distances[i, k] = dist

        #cluster_labels = np.zeros(samples, dtype='int')

        # basically just a self written argmin
        for i in range(samples):
            # track keeping variable for finding the smallest value in each column
            # set to big number to avoid missing numbers
            smallest = 1e10
            smallest_row_index = 1e10
            for k in range(n_clusters):
                if distances[i, k] < smallest:
                    smallest = distances[i, k]
                    smallest_row_index = k

            cluster_labels[i] = smallest_row_index

        centroid_difference = np.sum(np.abs(centroids - prev_centroids))

        if centroid_difference < tolerance:

            return cluster_labels, centroids

    print(f'Did not converge in {max_iterations} iterations')
    return cluster_labels, centroids



if __name__=='__main__':
    np.random.seed(2021)
    data = generate_simple_clustering_dataset(dim=2, n_points=1000, plotting=False, return_data=True)
    cluster_labels, centroids = k_means(data, max_iterations=100)
