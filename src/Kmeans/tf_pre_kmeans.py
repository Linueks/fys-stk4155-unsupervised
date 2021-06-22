"""
This is an adaptation of the kmeans example code from tensorflow which can be
found here: https://www.tensorflow.org/api_docs/python/tf/compat/v1/estimator/experimental/KMeans
Some changes to that script were made according to deprecations and readability
"""
import matplotlib.pyplot as plt
import numpy as np
from generate_dataset import *
from silence_tf import tensorflow_shutup
tensorflow_shutup()
import tensorflow as tf



def tf_pre_kmeans(data, n_clusters=4, max_iterations=20, tolerance=1e-8):

    def input_function():
        num_epochs = 1
        return tf.data.Dataset.from_tensors(
            tf.convert_to_tensor(data, dtype=tf.float32)).repeat(num_epochs)


    # initialize model
    kmeans = tf.compat.v1.estimator.experimental.KMeans(
                num_clusters=n_clusters,
                use_mini_batch=False,
                relative_tolerance=tolerance)

    # train model
    previous_centroids = None
    centroid_difference = 1e8
    for iteration in range(max_iterations):
        kmeans.train(input_function)
        centroids = kmeans.cluster_centers()

        if previous_centroids is not None:
            print(f'delta: {centroids - previous_centroids}')
            centroid_difference = np.sum(np.abs(centroids - previous_centroids))

        if centroid_difference < tolerance:
            cluster_labels = np.asarray(list(kmeans.predict_cluster_index(input_function)))
            print(f'Converged at iteration: {iteration}')
            return cluster_labels

        previous_centroids = centroids

    cluster_labels = np.asarray(list(kmeans.predict_cluster_index(input_function)))
    print(f'Did not converge in {max_iterations} iterations')

    return cluster_labels


if __name__=='__main__':
    np.random.seed(2021)
    tf.experimental.numpy.random.seed(2021)
    data = generate_clustering_dataset(dim=2, n_points=1000, plotting=False, return_data=True)
    cluster_labels = tf_pre_kmeans(data, max_iterations=20)
    unique_cluster_labels = np.unique(cluster_labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in unique_cluster_labels:
        ax.scatter(data[cluster_labels == i, 0],
                    data[cluster_labels == i, 1],
                    label = i)

    ax.set_title("Final Grouping")
    fig.tight_layout()
    plt.show()
