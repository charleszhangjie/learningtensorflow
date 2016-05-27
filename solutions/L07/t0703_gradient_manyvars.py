# From: https://github.com/nlintz/TensorFlow-Tutorials/blob/master/1_linear_regression.py
import tensorflow as tf
import numpy as np

x = tf.placeholder("float")
y = tf.placeholder("float")


def model(X, a, b):
    return tf.mul(X, a) + b # lr is just X*[a.b] so this model line is pretty simple

a = tf.Variable(1.0, name='a')
b = tf.Variable(2.0, name='b')

y_model = model(x, a, b)

cost = tf.square(y - y_model)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # construct an optimizer to minimize cost and fit line to my data
model = tf.initialize_all_variables()

with tf.Session() as session:
    session.run(model)
    for i in range(1000):
        x_value = np.random.rand(50)
        y_value = x_value * 2 + 6
        #print(x_value, y_value)
        session.run(train_op, feed_dict={x: x_value, y: y_value})
    print(session.run([a, b]))  # something around 2, 6