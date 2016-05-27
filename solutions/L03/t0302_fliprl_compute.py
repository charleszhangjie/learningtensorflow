import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.png"
image = mpimg.imread(filename)
# height, width, depth = image.shape

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
shape = tf.Variable(np.ones(3), name="shape")

model = tf.initialize_all_variables()

with tf.Session() as session:
    shape = tf.shape(x)
    session.run(model)
    shape = session.run(shape)

    # Try `tf.ones_initializer` to make this more robust!
    x = tf.reverse_sequence(x, np.ones((shape[0],)) * shape[1], 1, batch_dim=0)
    session.run(model)
    result = session.run(x)

print(result.shape)
plt.imshow(result)
plt.show()