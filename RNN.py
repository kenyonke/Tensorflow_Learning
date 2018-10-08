import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


#RNN
feature_size = 1
label_size = 1
time_step = 10

x = tf.placeholder(tf.float32,[None, time_step, feature_size])
y = tf.placeholder(tf.float32,[None, time_step, label_size])

rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = 32)
init = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)

# 'outputs' is a tensor of shape [batch_size, max_time, cell_state_size]
outputs,state = tf.nn.dynamic_rnn(
    rnn_cell,                # cell chosen
    x,                       # input
    #initial_state=None
    initial_state=init,  
    time_major=False, 
    )
outs2D = tf.reshape(outputs, [-1, 32]) 
net_outs2D = tf.layers.dense(outs2D, label_size)
outs = tf.reshape(net_outs2D, [-1, time_step, label_size]) 

loss = tf.losses.mean_squared_error(labels=y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph

plt.figure(1, figsize=(12, 5))
plt.ion()       # continuously plot

for step in range(80):
    start = step * np.pi
    end = (step+1)*np.pi
    # use sin predicts cos
    steps = np.linspace(start, end, time_step)
    # x is the input but we need show time steps which is steps in ploting 
    # np.newaxis: insert new dimension
    # Also, b[np.newaxis] is equals to b[np.newaxis,:]
    train_x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
    train_y = np.cos(steps)[np.newaxis, :, np.newaxis]
    
    if 'last_state' not in globals():                 # first state, no any hidden state
        feed_dict = {x: train_x, y: train_y}
    else:                                           # has hidden state, so pass it to rnn
        feed_dict = {x: train_x, y: train_y, init: last_state}
    _, pred_, last_state = sess.run([train_op, outs, state], feed_dict)     # train

    # plotting
    plt.plot(steps, train_y.flatten(), 'r-')
    plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2))
    plt.draw()

plt.ioff()
plt.show()
