import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.png"
image = plt.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
model = tf.initialize_all_variables()

with tf.Session() as session:
    x = tf.transpose(x, perm=[1, 0, 2])
    x = tf.reverse_sequence(x, np.ones((width,)) * height, 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()