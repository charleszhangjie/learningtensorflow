
import tensorflow as tf

x = tf.placeholder("float", [None, 3])
y = x * 2

with tf.Session() as session:
    x_data = [[1, 2, 3],
              [4, 5, 6],]
    result = session.run(y, feed_dict={x: x_data})
    print(result)
    print(type(y))
    

"""
import tensorflow as tf

a = tf.placeholder("float") # Create a symbolic variable 'a'
b = tf.placeholder("float") # Create a symbolic variable 'b'

y = tf.mul(a, b) # multiply the symbolic variables

sess = tf.Session() # create a session to evaluate the symbolic expressions

print "%f should equal 2.0" % sess.run(y, feed_dict={a: 1, b: 2}) # eval expressions with parameters for a and b
print "%f should equal 9.0" % sess.run(y, feed_dict={a: 3, b: 3})
"""