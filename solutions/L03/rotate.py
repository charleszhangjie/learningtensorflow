import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)

# Create a TensorFlow Variable
x = tf.Variable(image, name='x')
y = tf.transpose(x, perm=[1, 0, 2])
z = tf.transpose(y, perm=[1, 0, 2])
model = tf.initialize_all_variables()

with tf.Session() as session:
    for i in range(2):
        #x = tf.transpose(x, perm=[1, 0, 2])
        session.run(model)
        result = session.run(z)

print(result.shape)
plt.imshow(result)
plt.show()
