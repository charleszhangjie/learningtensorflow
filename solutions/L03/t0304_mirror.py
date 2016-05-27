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
mirrored = tf.Variable(image, name='mirrored')
model = tf.initialize_all_variables()

mirror_mask = np.ones((height,)) * (width/2)


with tf.Session() as session:
    # Note swapped dims in the last two parameters
    mirrored = tf.reverse_sequence(x, mirror_mask, 1, batch_dim=0)

    # Now stich them back up again


    session.run(model)
    result = session.run(mirrored)

print(result.shape)
plt.imshow(result)
plt.show()