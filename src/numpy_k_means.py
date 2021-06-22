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
    distances = np.sum(np.abs((data - centroids))**2, axis=2)
    #distances = np.linalg.norm(data - centroids, axis=2)
    return distances



def assign_points_to_clusters(distances):
    samples, n_clusters = distances.shape
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels



def k_means(data, n_clusters=4, max_iterations=20, tolerance=1e-8, plot_results=True):
    centroids = data[np.random.choice(len(data), n_clusters, replace=False), :]

    distances = get_distances_to_clusters(data.reshape((4000, 1, 2)),
                                            centroids.reshape((1, 4, 2)))
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
        distances = get_distances_to_clusters(np.reshape(data, (4000, 1, 2)),
                                                np.reshape(centroids, (1, 4, 2)))
        cluster_labels = assign_points_to_clusters(distances)

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

    return cluster_labels, centroids_list


if __name__=='__main__':
    np.random.seed(2021)
    data = generate_clustering_dataset(dim=2, n_points=1000, plotting=False, return_data=True)
    k_means(data, max_iterations=20, plot_results=True)
