"""
In this file we want to implement a numpythonic version of the previous two
programs. Here we wish to vectorize loops that take a long time, mainly the
loops that go over the number of samples as this grows very large. Using the
cprofilev tool we can see which functions / loops are the biggest bottlenecks in
our program and subsequently work to try to vectorize these.
"""
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
            plot_results=True, progression_plot=True):
    """
    Numpythonic implementation of the k-means clusting algorithm.

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
    n_samples, dimensions = data.shape
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]

    distances = get_distances_to_clusters(data.reshape((n_samples, 1, dimensions)),
                                    centroids.reshape((1, n_clusters, dimensions)))
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

    centroids_list = []
    temp_centroids = centroids.copy()
    centroids_list.append(temp_centroids)


    for iteration in range(max_iterations):
        for k in range(n_clusters):
            points_in_cluster = data[cluster_labels == k]
            mean_vector = np.mean(points_in_cluster, axis=0)
            centroids[k] = mean_vector

        temp_centroids = centroids.copy()
        centroids_list.append(temp_centroids)
        distances = get_distances_to_clusters(np.reshape(data, (n_samples, 1, dimensions)),
                                                np.reshape(centroids, (1, n_clusters, dimensions)))
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
                        s = 8,
                        alpha = 0.6)

                ax.scatter(centroids[:, 0], centroids[:, 1], c='black')

                ax.set_title(f'Clusters at iteration {iteration}')
                #fig.tight_layout()
                #fig.legend()
                plt.savefig(file_folder + f'c_image_at_it_{str(iteration).zfill(3)}.png')
            plt.close()




        centroid_difference = np.sum(np.abs(centroids_list[iteration] - centroids_list[iteration-1]))
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

            return cluster_labels, centroids_list

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
    return cluster_labels, centroids_list


if __name__=='__main__':
    np.random.seed(2021)
    simple_data = generate_simple_clustering_dataset(n_points=1000, plotting=False)
    complicated_data = generate_complicated_clustering_dataset(n_points=1000, plotting=False)
    k_means(complicated_data, n_clusters=8, plot_results=False, progression_plot=True)
