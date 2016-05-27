import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")

from matplotlib import pyplot as plt

plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

import matplotlib.animation as animation
from scipy.signal import convolve2d


# Update to 0.7+ with a nightly build
# Get the latest nightly for your system at https://github.com/tensorflow/tensorflow#installation
# Install wheel files with  python -m wheel install ~/Downloads/tensorflow-0.7.1-cp34-cp34m-linux_x86_64.whl


def update_board(X):
    # Check out the details at: https://jakevdp.github.io/blog/2013/08/07/conways-game-of-life/
    # Compute number of neighbours,
    N = convolve2d(X, np.ones((3, 3)), mode='same', boundary='wrap') - X
    # Apply rules of the game
    X = (N == 3) | (X & (N == 2))
    return X


shape = (50, 50)
initial_board = tf.random_uniform(shape, minval=0, maxval=2, dtype=tf.int32, name='initial_board')

board = tf.placeholder(tf.int32, shape=shape, name='board')
board_update = tf.py_func(update_board, [board], [tf.int32])


fig = plt.figure()

loop = 0

with tf.Session() as session:
    X = session.run(initial_board)
    plot = plt.imshow(X, cmap='Greys',  interpolation='nearest')


    def game_of_life(*args):
        global X, loop
        # Stop after 50 iterations
        if loop < 50:
            loop += 1
            X = session.run(board_update, feed_dict={board: X})[0]
        plot.set_array(X)
        return plot,

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani = animation.FuncAnimation(fig, game_of_life, interval=200, blit=True)
    #ani.save('game.mp4', writer=writer)
    plt.show()
