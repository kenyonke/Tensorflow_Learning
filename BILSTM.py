# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 02:23:08 2018

@author: kenyon
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''定义LSTM模型'''
class lstm_model:
    def __init__(self):
        HIDDEN_SIZE = 20
        NUM_LAYERS = 1
        keep_prob = 1
        INPUT_SIZE = 1 #batch_size
        TIME_STEP = 10
        
        #define placeholder
        self.x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape(batch, 10, 1)
        self.y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])   
        
        #定义bi-LSTM cell
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        
        #dropout
        lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_fw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_bw_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
        
        #定义多层bi-LSTM cell
        lstm_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell for _ in range(NUM_LAYERS)])
        lstm_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell for _ in range(NUM_LAYERS)])

        #state中的hidden size已经在创建cell时定义好了，这里初始化需要输入batch size用来记录每个batch所输出的final_state(给下一batch的当作输入吧？！大概)
        #因此就需要[batch_size，hidden_size]来存储整个batch每个seq的状态
        self.init_fw = lstm_fw_cell.zero_state(INPUT_SIZE, dtype=tf.float32) #all zero initialized state
        self.init_bw = lstm_bw_cell.zero_state(INPUT_SIZE, dtype=tf.float32) #all zero initialized state
        
        '''将已经堆叠起的LSTM单元转化成动态的可在训练过程中更新的LSTM单元'''
        outputs, self.final_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,
                                                        lstm_bw_cell,
                                                        self.x,
                                                        initial_state_fw = self.init_fw,
                                                        initial_state_bw = self.init_bw)
        print(outputs[0].shape) # the shape of outpus now is 2*10*20  (2states*time_state*hidden_size)
        outputs = tf.concat(outputs, axis=2)   #将前向和后向的状态连接起来
        print(outputs.shape)
        '''根据预定义的每层神经元个数来生成隐层每个单元'''
        outputs = tf.reshape(outputs, [-1, HIDDEN_SIZE*2])
        
        '''通过无激活函数的全连接层计算线性回归，并将数据压缩成一维数组结构'''
        net_outs2D = tf.layers.dense(outputs, INPUT_SIZE)
    
        '''统一预测值与真实值的形状'''
        self.predictions = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])
    
        '''定义损失函数，这里为正常的均方误差'''
        self.loss = tf.losses.mean_squared_error(self.predictions, self.y)
        
        self.train_op = tf.train.AdamOptimizer(learning_rate=0.02).minimize(self.loss)
        
if __name__ == '__main__':
    
    plt.figure(1, figsize=(12, 5))
    plt.ion()       # continuously plot
    lstm = lstm_model()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for step in range(80):
            start = step * np.pi
            end = (step+1)*np.pi
            TIME_STEP = 10
            # use sin predicts cos
            steps = np.linspace(start, end, TIME_STEP)
            # x is the input but we need show time steps which is steps in ploting 
            x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
            y = np.cos(steps)[np.newaxis, :, np.newaxis]
            
            if 'last_state' not in globals():                 # first state, no any hidden state
                feed_dict = {lstm.x: x, lstm.y: y}
            else:                                           # has hidden state, so pass it to rnn
                feed_dict = {lstm.x: x, lstm.y: y, lstm.init_fw:last_state[0], lstm.init_bw:last_state[1] }
            _, pred_,last_state = sess.run([lstm.train_op, lstm.predictions, lstm.final_state], feed_dict)     # train            
            
            # plotting
            plt.plot(steps, y.flatten(), 'r-')
            plt.plot(steps, pred_.flatten(), 'b-')
            plt.ylim((-1.2, 1.2))
            plt.draw()
            
