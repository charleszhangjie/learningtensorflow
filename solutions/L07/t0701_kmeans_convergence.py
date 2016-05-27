import tensorflow as tf
import numpy as np

from L06.functions import *


n_features = 2
n_clusters = 3
n_samples_per_cluster = 500
seed = 700
embiggen_factor = 70

min_change = 10

data_centroids, samples = create_samples(n_clusters, n_samples_per_cluster, n_features, embiggen_factor, seed)
centroids = choose_random_centroids(samples, n_clusters, seed=seed)
old_centroids = None

model = tf.initialize_all_variables()
with tf.Session() as session:
    session.run(samples)
    while True:
        nearest_indices = assign_to_nearest(samples, centroids)
        centroids = update_centroids(samples, nearest_indices, n_clusters)
        centroid_values = session.run(centroids)
        # There is another great convergence example at https://gist.github.com/dave-andersen/265e68a5e879b5540ebc
        if old_centroids is not None:
            print(np.sum(np.abs(old_centroids - centroid_values), axis=None))
        if old_centroids is not None and np.sum(np.abs(old_centroids - centroid_values)) < min_change:
            break
        old_centroids = centroid_values
    sample_values = session.run(samples)
    updated_centroid_value = session.run(centroids)
    print(updated_centroid_value)

plot_clusters(sample_values, updated_centroid_value, n_samples_per_cluster)