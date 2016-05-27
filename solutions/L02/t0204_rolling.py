# Exercise 4

import tensorflow as tf
import numpy as np


mean = tf.Variable(0., name='mean')
n = tf.Variable(0., name='n')

model = tf.initialize_all_variables()


m = 10000

with tf.Session() as session:
    for i in range(5):
        new_random_numbers = np.random.randint(1000, size=m)
        sum_of_random_numbers = np.sum(new_random_numbers)
        
        n += m
        
        mean = (mean * (n-m)/n) + (sum_of_random_numbers / n)
  
    session.run(model)
    print(session.run(mean))

