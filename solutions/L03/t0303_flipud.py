import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.png"
image = mpimg.imread(filename)
height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')

model = tf.initialize_all_variables()

with tf.Session() as session:
    # Note swapped dims in the last two parameters
    x = tf.reverse_sequence(x, np.ones((width,)) * height, 0, batch_dim=1)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()