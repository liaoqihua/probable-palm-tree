# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 15:43:23 2018

@author: LQH
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)

mnist = input_data.read_data_sets("E:\Python\MNIST\mnist", one_hot=True)
print("train images shape", mnist.train.images.shape)
print("train labels shape", mnist.train.labels.shape)

def get_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def get_biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

#fc1
w_fc1 = get_weights([784, 1024])
b_fc1 = get_biases([1024])
h_fc1 = tf.nn.relu(tf.matmul(X, w_fc1) + b_fc1)

#fc2
w_fc2 = get_weights([1024, 10])
b_fc2 = get_biases([10])
h_pre = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(Y * tf.log(h_pre))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h_pre, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(5000):
        x_batch, y_batch = mnist.train.next_batch(batch_size=128)
        sess.run([train_step], feed_dict={X:x_batch, Y:y_batch})
        
        if (i+1) % 200 == 0:
            acc = sess.run(accuracy, feed_dict={X:mnist.train.images, Y:mnist.train.labels})
            print("step {0}, train acc is {1}".format((i+1), acc))
            
        if (i+1) % 500 == 0:
            acc = sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels})
            print("step {0}, test acc is {1}".format((i+1), acc))
'''

sess.run(tf.global_variables_initializer())
    
for i in range(5000):
    x_batch, y_batch = mnist.train.next_batch(batch_size=128)
    train_step.run(feed_dict={X:x_batch, Y:y_batch})
    
    if (i+1) % 200 == 0:
        acc = accuracy.eval(feed_dict={X:mnist.train.images, Y:mnist.train.labels})
        print("step {0}, train acc is {1}".format((i+1), acc))
        
    if (i+1) % 500 == 0:
        acc = accuracy.eval(feed_dict={X:mnist.test.images, Y:mnist.test.labels})
        print("step {0}, test acc is {1}".format((i+1), acc))
        
sess.close()