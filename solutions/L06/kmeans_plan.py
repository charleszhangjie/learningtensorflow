

# Create random dataset of blogs


# Use "tf.gather" with a randomly generated index to initialise starting centroids

# Loop until convergence <<< TODO convergence test
# Start with just 100 iterations, then consider a convergence test later on


# Step 1: Assignment
# Compute distance from each point to each centroid

# use tf.argmin https://www.tensorflow.org/versions/0.6.0/api_docs/python/math_ops.html#argmin
# to get centroid assignments

# To compute distance, use something like:
# euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.sub(v1, v2), 2)))
# This line from https://codesachin.wordpress.com/2015/11/14/k-means-clustering-with-tensorflow/
# distances = [sess.run(euclid_dist, feed_dict={v1: vect, v2: sess.run(centroid)}) for centroid in centroids]

# Step 2: Optimisation
# Use tf.gather to get all points associated with a particular centroid
# use tf.reduce_mean https://www.tensorflow.org/versions/0.6.0/api_docs/python/math_ops.html#reduce_mean
# To compute new centroid

# Can also use tf.segment_sum to do this https://www.tensorflow.org/versions/0.6.0/api_docs/python/math_ops.html#segment_sum
# However that is what the reference uses