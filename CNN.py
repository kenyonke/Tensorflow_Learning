# -*- coding: utf-8 -*-
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#load minist image data
mnist = input_data.read_data_sets('.\mnist', one_hot=True)  # they has been normalized to range (0,1)
train_x = mnist.train.images[:4000]
train_y = mnist.train.labels[:4000]

test_x = mnist.test.images[4000:4300]
test_y = mnist.test.labels[4000:4300]

#Convolutional Nerual Network model
data_x = tf.placeholder(tf.float32,[None,784])
data_y = tf.placeholder(tf.int32,[None,10])

x = tf.reshape(data_x, [-1, 28, 28, 1])  

conv1 = tf.layers.conv2d(
        inputs = x,
        filters = 10,
        kernel_size = 5,
        strides = 1,
        padding = 'same',
        activation = tf.nn.rtelu
        ) 

pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = 2,
        strides = 2
        )

conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2,2,2)
cnn_output = tf.reshape(pool2, [-1, 7*7*32])

w = tf.Variable(tf.zeros(shape=[7*7*32,10]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros(shape=[10]),dtype=tf.float32,name='b')

pred = tf.nn.softmax(tf.add(tf.matmul(cnn_output,w),b))
loss = tf.losses.softmax_cross_entropy(data_y,pred)
train = tf.train.AdamOptimizer(0.01).minimize(loss)
acc = tf.metrics.accuracy(labels=tf.argmax(data_y, axis=1), predictions=tf.argmax(pred, axis=1),)[1]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())   
    
    for i in range(200):
        #batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE=200)  #batch training data
        loss_val,_ = sess.run([loss,train],feed_dict = {data_x:train_x,data_y:train_y})
        print(loss_val)
    
    accuracy = sess.run([acc],feed_dict = {data_x:test_x,data_y:test_y})
    print('accuracy:',accuracy)    