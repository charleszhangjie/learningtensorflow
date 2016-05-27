import tensorflow as tf

import matplotlib.image as mpimg

# First, load the image
filename = "MarshOrchid.jpg"
image = mpimg.imread(filename)
print(image.shape)
print(type(image))

#plt.interactive(False)
import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()
