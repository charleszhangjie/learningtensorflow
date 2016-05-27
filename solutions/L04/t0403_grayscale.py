import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.png"
raw_image_data = mpimg.imread(filename)
print(raw_image_data.shape, raw_image_data.dtype)
height, width, depth = raw_image_data.shape

image = tf.placeholder("float32", [None, None, 3])
grayscale = tf.reduce_mean(image, 2)  # Compute mean along last axis

with tf.Session() as session:
    result = session.run(grayscale, feed_dict={image: raw_image_data})
    print(result.shape)

    plt.imshow(result, cmap="gray")
    plt.show()
