import tensorflow as tf
import numpy as np

x = tf.constant(np.random.randint(1000, size=10000), name='x')
y = tf.Variable((5 * x ** 2) - (3 * x) + 15, name='y')


model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    print(session.run(y))

