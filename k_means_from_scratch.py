import numpy as np
import matplotlib.pyplot as plt
#plt.style.use('ggplot')
# Program naively implementing k-means algorithm from scratch in Numpy Python


def gaussian_points(dim=2, n_points=100, mean_vector=[0, 0], sample_variance=1):
    """
    Very simple custom function to generate gaussian distributed point clusters
    with variable dimension, number of points, means in each direction
    (must match dim) and sample variance.

    dim: float
    n_points: int
    mean_vector: list (where index 0 is x, index 1 is y and so on) must match dim
    sample_variance: float
    """

    mean_matrix = np.zeros(dim) + np.asarray(mean_vector)
    covariance_matrix = np.eye(dim) * sample_variance
    data = np.random.multivariate_normal(mean_matrix, covariance_matrix, n_points)
    return data.T



def generate_clustersing_dataset(plotting=True, return_data=True):
    """
    Toy model to illustrate k-means clustering
    """

    x1, y1 = gaussian_points(mean_vector=[5, 5])
    x2, y2 = gaussian_points()
    x3, y3 = gaussian_points(mean_vector=[1, 4.5])
    x4, y4 = gaussian_points(mean_vector=[5, 1])

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
        return data_x, data_y



def k_means(data, n_clusters=4, max_iterations=20):
    # first we collect our 'dataset' in one array for convenience

    # we need to (randomly) choose initial centroids
    centroids = np.zeros((data.shape[0], n_clusters))
    for k in range(n_clusters):
        idx = np.random.randint(0, data.shape[1])                               # TODO: There is one bug here that with very few points two centroids might be the same point
        centroids[:, k] = data[0, idx], data[1, idx]

    # next we initialize an array to hold the distance from each centroid to
    # every point in our dataset (There are ways to optimize, we'll get to that)
    distances = np.zeros((n_clusters, data.shape[1]))

    # we find the squared Euclidean distance from each centroid to every point
    for k in range(n_clusters):
        for i in range(data.shape[1]):
            dist = 0
            for j in range(data.shape[0]):
                dist += np.abs(data[j, i] - centroids[j, k])**2
                distances[k, i] = dist
                # the way this is setup now we have Dist[i, j] = the distance between the i-th
                # point and the j-th cluster

    # then we need to assign each data point to their closest centroid
    # based on squared Euclidean distance
    # This can probably be done in one loop but starting with simplest for me


    # We initialize a list to put our points into clusters. This way we do it here
    # the index signifies which point and the value which cluster
    cluster_labels = np.zeros(data.shape[1], dtype='int')

    # basically just a self written argmin
    for i in range(distances.shape[1]):
        # track keeping variable for finding the smallest value in each column
        # set to big number to avoid missing numbers
        smallest = 1e10
        smallest_row_index = 1e10
        for k in range(distances.shape[0]):
            #print(distances[k, i])
            if distances[k, i] < smallest:
                smallest = distances[k, i]
                smallest_row_index = k

        cluster_labels[i] = smallest_row_index

    debug_plot = False
    if debug_plot:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        axs[0, 0].scatter(data[0], data[1])
        axs[0, 0].set_title("Toy Model Dataset")


        axs[0, 1].scatter(data[0], data[1])
        axs[0, 1].scatter(centroids[0, :], centroids[1, :])
        axs[0, 1].set_title("Initial Random Centroids")


        unique_cluster_labels = np.unique(cluster_labels)
        for i in unique_cluster_labels:
            axs[1, 0].scatter(data[0, cluster_labels == i] , data[1, cluster_labels == i] , label = i)

        axs[1, 0].set_title("First Grouping of Points to Centroids")
        plt.tight_layout()
        plt.show()


    centroid_list = []
    centroid_list.append(centroids)

    # For each cluster we need to update the centroid by calculating new means
    # for all the data points in the cluster and repeat
    for _ in range(max_iterations):
        for k in range(n_clusters):
            # Here we find the mean for each centroid
            vector_mean = np.zeros(data.shape[0])
            n = 0
            for i in range(data.shape[1]):
                if cluster_labels[i] == k:
                    vector_mean += data[:, i]
                    n += 1
            # And update according to the new means
            centroids[:, k] = vector_mean / n

            distances = np.zeros((n_clusters, data.shape[1]))

        centroid_list.append(centroids)
        # we find the squared Euclidean distance from each centroid to every point
        for k in range(n_clusters):
            for i in range(data.shape[1]):
                dist = 0
                for j in range(data.shape[0]):
                    dist += np.abs(data[j, i] - centroids[j, k])**2
                    distances[k, i] = dist

        cluster_labels = np.zeros(data.shape[1], dtype='int')

        # basically just a self written argmin
        for i in range(distances.shape[1]):
            # track keeping variable for finding the smallest value in each column
            # set to big number to avoid missing numbers
            smallest = 1e10
            smallest_row_index = 1e10
            for k in range(distances.shape[0]):
                #print(distances[k, i])
                if distances[k, i] < smallest:
                    smallest = distances[k, i]
                    smallest_row_index = k

            cluster_labels[i] = smallest_row_index


    return cluster_labels





print("testing")

if __name__=='__main__':
    np.random.seed(2021)
    x, y = generate_clustersing_dataset(plotting=True)
    data = np.array([x, y])

    #x = [1, 4, 3, 10, 5]
    #y = [11, 9, 8, 10, 15]

    labels = k_means(data)


    unique_cluster_labels = np.unique(labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in unique_cluster_labels:
        ax.scatter(data[0, labels == i] , data[1, labels == i] , label = i)

    ax.set_title("Final Grouping")
    plt.tight_layout()
    plt.show()
