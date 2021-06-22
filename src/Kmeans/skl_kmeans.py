import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from generate_dataset import *



if __name__=='__main__':
    np.random.seed(2021)
    data = generate_clustering_dataset(dim=2, n_points=1000, plotting=False, return_data=True)
    kmeans = KMeans(n_clusters=4, max_iter=20, tol=1e-8).fit(data)

    cluster_labels = kmeans.labels_
    unique_cluster_labels = np.unique(cluster_labels)
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in unique_cluster_labels:
        ax.scatter(data[cluster_labels == i, 0],
                    data[cluster_labels == i, 1],
                    label = i)

    ax.set_title("Final Grouping")
    fig.tight_layout()
    plt.show()
