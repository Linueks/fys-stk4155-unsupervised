"""
A simple toy model I quickly made to test my implementation of the k-means
clusting algorithm. This basically does the same as make_blobs in sklearn, but
I found that functionality after I was already using this so..
--------------------------------------------------------------------------------
Written Summer 2021 Linus Ekstrom for FYS-STK4155 course content. A lot like the
"""
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
    data = np.random.multivariate_normal(mean_matrix, covariance_matrix,
                                    n_points)
    return data



def generate_simple_clustering_dataset(dim=2, n_points=1000, plotting=True,
                                    return_data=True):
    """
    Toy model to illustrate k-means clustering
    """

    data1 = gaussian_points(mean_vector=np.array([5, 5]))
    data2 = gaussian_points()
    data3 = gaussian_points(mean_vector=np.array([1, 4.5]))
    data4 = gaussian_points(mean_vector=np.array([5, 1]))
    data = np.concatenate((data1, data2, data3, data4), axis=0)

    if plotting:
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.scatter(data[:, 0], data[:, 1])
        ax1.set_aspect('equal')
        #ax1.grid()

        plt.show()


    if return_data:
        return data


def generate_complicated_clustering_dataset(dim=2, n_points=1000, plotting=True, return_data=True):
    """
    Another toy model
    """

    data1 = gaussian_points(mean_vector=np.array([2.5, 0]))
    data2 = gaussian_points(mean_vector=np.array([0, 7.5]))
    data3 = gaussian_points(mean_vector=np.array([4, 4]))
    data4 = gaussian_points(mean_vector=np.array([2.5, -6]))
    data5 = gaussian_points(mean_vector=np.array([5, -2]))
    data6 = gaussian_points(mean_vector=np.array([-5, 3]))
    data7 = gaussian_points(mean_vector=np.array([-1, -1]))
    data8 = gaussian_points(mean_vector=np.array([-4, -4]))
    data9 = gaussian_points(mean_vector=np.array([8, -4]))

    data = np.concatenate((data1, data2, data3, data4, data5, data6, data7,
                        data8, data9), axis=0)

    if plotting:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot()
        #fig, ax = plt.subplots(figsize=(8, 5), projection='3d')
        ax.scatter(data[:, 0], data[:, 1])
        plt.show()

    if return_data:
        return data




if __name__=='__main__':

    data = generate_complicated_clustering_dataset()
