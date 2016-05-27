import numpy as np

from matplotlib import pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

X = np.random.randint(0, 2, (50, 50))

C = np.ones((3, 3))
loop = 0  # loop number
fig = plt.figure()
plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')

def game_of_life(*args):
    global X, prev_X, loop
    loop += 1
    prev_X = X
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    plot.set_array(X)
    # Stop if the grid hasn't changed
    if (X == prev_X).all():
        print("Finished on loop {}. Press CTRL+C (or close the plot window) to exit".format(loop))
        raise StopIteration()
    return plot,

ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)


plt.show()