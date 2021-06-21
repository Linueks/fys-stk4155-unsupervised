"""
Program naively implementing k-means algorithm from scratch in Python
Written Summer 2021 Linus Ekstrom for FYS-STK4155 course content
This file is the absolute 'simplest' implementation I could do of k-means.
However, I feel some clarity is lost when not using functions whether custom
or from various libraries.
"""
import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')



def gaussian_points(dim=2, n_points=1000, mean_vector=np.array([0, 0]),
                    sample_variance=1):
    """
    Very simple custom function to generate gaussian distributed point clusters
    with variable dimension, number of points, means in each direction
    (must match dim) and sample variance.

    Inputs:
        dim (int)
        n_points (int)
        mean_vector (np.array) (where index 0 is x, index 1 is y etc.)
        sample_variance (float)

    Returns:
        data (np.array): with dimensions (dim x n_points)
    """


    mean_matrix = np.zeros(dim) + mean_vector
    covariance_matrix = np.eye(dim) * sample_variance
    data = np.random.multivariate_normal(mean_matrix, covariance_matrix, n_points)
    return data.T



def generate_clustering_dataset(plotting=True, return_data=True):
    """
    Toy model to illustrate k-means clustering
    """

    x1, y1 = gaussian_points(mean_vector=np.array([5, 5]))
    x2, y2 = gaussian_points()
    x3, y3 = gaussian_points(mean_vector=np.array([1, 4.5]))
    x4, y4 = gaussian_points(mean_vector=np.array([5, 1]))

    data_x = np.concatenate([x1, x2, x3, x4])
    data_y = np.concatenate([y1, y2, y3, y4])

    if plotting:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        fig2, ax2 = plt.subplots(figsize=(8, 5))

        ax1.scatter(x1, y1)
        ax1.scatter(x2, y2)
        ax1.scatter(x3, y3)
        ax1.scatter(x4, y4)
        ax1.set_aspect('equal')
        #ax1.grid()

        ax2.scatter(data_x, data_y)
        ax2.set_aspect('equal')
        #ax2.grid()

        plt.show()


    if return_data:
        data = np.array([data_x, data_y])
        return data



def k_means(data, n_clusters=4, max_iterations=20, tolerance=1e-8,
            debug_plot=False):

    dimensions, samples = data.shape
    # we need to (randomly) choose initial centroids
    centroids = np.zeros((dimensions, n_clusters))
    for k in range(n_clusters):
        idx = np.random.randint(0, samples)                                     # TODO: There is one bug here that with very few points two centroids might be the same point
        centroids[:, k] = data[0, idx], data[1, idx]

    # next we initialize an array to hold the distance from each centroid to
    # every point in our dataset
    distances = np.zeros((n_clusters, samples))
    # we find the squared Euclidean distance from each centroid to every point
    for k in range(n_clusters):
        for i in range(samples):
            dist = 0
            for j in range(dimensions):
                dist += np.abs(data[j, i] - centroids[j, k])**2
                distances[k, i] = dist
                # the way this is setup now we have Dist[i, j] = the distance between the i-th
                # point and the j-th cluster

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
            if distances[k, i] < smallest:
                smallest = distances[k, i]
                smallest_row_index = k

        cluster_labels[i] = smallest_row_index

    if debug_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].scatter(data[0], data[1])
        axs[0, 0].set_title("Toy Model Dataset")


        axs[0, 1].scatter(data[0], data[1])
        axs[0, 1].scatter(centroids[0, :], centroids[1, :])
        axs[0, 1].set_title("Initial Random Centroids")


        unique_cluster_labels = np.unique(cluster_labels)
        for i in unique_cluster_labels:
            axs[1, 0].scatter(data[0, cluster_labels == i],
                                data[1, cluster_labels == i],
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
                    vector_mean += data[:, i]
                    n += 1
            # And update according to the new means
            centroids[:, k] = vector_mean / n
            distances = np.zeros((n_clusters, samples))

        # we need to use copies to avoid overwriting (pointer stuff)
        temp_centroids = centroids.copy()
        centroid_list.append(temp_centroids)

        # we find the squared Euclidean distance from each centroid to every point
        for k in range(n_clusters):
            for i in range(samples):
                dist = 0
                for j in range(dimensions):
                    dist += np.abs(data[j, i] - centroids[j, k])**2
                    distances[k, i] = dist

        cluster_labels = np.zeros(samples, dtype='int')

        # basically just a self written argmin
        for i in range(samples):
            # track keeping variable for finding the smallest value in each column
            # set to big number to avoid missing numbers
            smallest = 1e10
            smallest_row_index = 1e10
            for k in range(n_clusters):
                if distances[k, i] < smallest:
                    smallest = distances[k, i]
                    smallest_row_index = k

            cluster_labels[i] = smallest_row_index

        centroid_difference = np.sum(np.abs(centroid_list[iteration] - centroid_list[iteration-1]))

        if centroid_difference < tolerance:
            if debug_plot:
                unique_cluster_labels = np.unique(cluster_labels)
                for i in unique_cluster_labels:
                    axs[1, 1].scatter(data[0, cluster_labels == i],
                                data[1, cluster_labels == i],
                                label = i)

                axs[1, 1].set_title("Final Grouping")
                fig.tight_layout()
                plt.show()

            print(f'Converged at iteration: {iteration}')
            return cluster_labels, centroid_list

    if debug_plot:
        unique_cluster_labels = np.unique(cluster_labels)
        for i in unique_cluster_labels:
            axs[1, 1].scatter(data[0, cluster_labels == i],
                        data[1, cluster_labels == i],
                        label = i)

        axs[1, 1].set_title("Final Grouping")
        fig.tight_layout()
        plt.show()

    return cluster_labels, centroid_list



if __name__=='__main__':
    #np.random.seed(2021)
    x, y = generate_clustering_dataset(plotting=False)
    cluster_labels, centroids = k_means(data, debug_plot=True)
