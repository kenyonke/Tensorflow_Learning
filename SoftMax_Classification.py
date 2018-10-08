import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

#load minist image data
mnist = input_data.read_data_sets('.\mnist', one_hot=True)  # they has been normalized to range (0,1)
train_x = mnist.train.images[:4000]
train_y = mnist.train.labels[:4000]

test_x = mnist.test.images[4000:4300]
test_y = mnist.test.labels[4000:4300]

data_x = tf.placeholder(tf.float32,[None,784])
data_y = tf.placeholder(tf.int32,[None,10])

#softmax model
w = tf.Variable(tf.zeros(shape=[784,10]), dtype=tf.float32, name='w')
b = tf.Variable(tf.zeros(shape=[10]),dtype=tf.float32,name='b')

pred = tf.nn.softmax(tf.add(tf.matmul(data_x,w),b))
loss = tf.losses.softmax_cross_entropy(data_y,pred)
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
acc = tf.metrics.accuracy(labels=tf.argmax(data_y, axis=1), predictions=tf.argmax(pred, axis=1),)[1]

#training & testing
with tf.Session() as sess:
    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    for i in range(500):
        loss_val,_ = sess.run([loss,train],feed_dict = {data_x:train_x,data_y:train_y})
        print(loss_val)
    
    accuracy = sess.run([acc],feed_dict = {data_x:test_x,data_y:test_y})
    print('accuracy:',accuracy)