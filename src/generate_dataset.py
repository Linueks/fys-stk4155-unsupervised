import numpy as np
import matplotlib.pyplot as plt



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
