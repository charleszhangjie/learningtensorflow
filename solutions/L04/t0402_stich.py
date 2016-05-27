import tensorflow as tf
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# First, load the image again
filename = "MarshOrchid.png"
raw_image_data = mpimg.imread(filename)
print(raw_image_data.shape, raw_image_data.dtype)
height, width, depth = raw_image_data.shape

image = tf.placeholder("float32", [None, None, 3])
topleft = tf.slice(image, [0, 0, 0], [height//2, width//2, -1])
bottomleft = tf.slice(image, [height//2, 0, 0], [-1, width//2, -1])
topright = tf.slice(image, [0, width//2, 0], [height//2, -1, -1])
bottomright = tf.slice(image, [height//2, width//2, 0], [-1, -1, -1])


top_stich = tf.concat(1, [topleft, topright], name='topstich')
bottom_stich = tf.concat(1, [bottomleft, bottomright], name='bottomstich')
stiched = tf.concat(0, [top_stich, bottom_stich], name='concat')

with tf.Session() as session:
    result = session.run(topleft, feed_dict={image: raw_image_data})
    print(result.shape)

    if False:
        # Show each of the corners as separate plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        ax1.imshow(session.run(topleft, feed_dict={image: raw_image_data}))
        ax2.imshow(session.run(topright, feed_dict={image: raw_image_data}))
        ax3.imshow(session.run(bottomleft, feed_dict={image: raw_image_data}))
        ax4.imshow(session.run(bottomright, feed_dict={image: raw_image_data}))
        plt.show()

    if False:
        # Show just the top stich
        plt.imshow(session.run(top_stich, feed_dict={image: raw_image_data}))
        plt.show()

    # Show the stiched up result
    result = session.run(stiched, feed_dict={image: raw_image_data})
    plt.imshow(result)
    plt.show()
