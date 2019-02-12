# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 01:57:16 2018

@author: kenyon
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
from tensorflow.examples.tutorials.mnist import input_data

class cgan:
    def __init__(self, z_size,batch_size):
        self.z_size = z_size
        self.batch_size = batch_size
        
        self.x = tf.placeholder(tf.float32,[None, 28, 28, 1],name='real_img')
        self.c = tf.placeholder(tf.float32,[None,10],name='one-hot')
        self.random_x = tf.placeholder(tf.float32,[None, 28, 28, 1],name='random_real_img')
        self.noise = tf.placeholder(tf.float32,[None,self.z_size],name='noise_input')
        
    def generator(self, noise, c, is_train=True, alpha=0.01, cannel_size=1):
        with tf.variable_scope('generator', reuse=not is_train):
            
            #concat noise and c
            noise = tf.concat([noise, c],axis=1)
            
            # input -> 4*4*512
            h1 = tf.reshape(tf.layers.dense(noise, 4*4*512), [-1,4,4,512])
            # batch normalization
            h1 = tf.layers.batch_normalization(h1, training=is_train)
            # Leaky Relu activate
            h1 = tf.maximum(alpha*h1, h1)
            # dropout
            h1 = tf.nn.dropout(h1, keep_prob = 0.8)
            
            # 4*4*512 -> 7*7*256
            h2 = tf.layers.conv2d_transpose(h1, filters = 256, kernel_size = 4, 
                                            strides=1, padding='valid')
            h2 = tf.layers.batch_normalization(h2,training=is_train)
            h2 = tf.maximum(alpha*h2, h2)
            h2 = tf.nn.dropout(h2, keep_prob = 0.8)
            
            #7*7*256 -> 14*14*128
            h3 = tf.layers.conv2d_transpose(h2, filters=128, kernel_size = 3,
                                            strides=2, padding='same')
            h3 = tf.layers.batch_normalization(h3, training=is_train)
            h3 = tf.maximum(alpha*h3, h3)
            h3 = tf.nn.dropout(h3, keep_prob = 0.8)
            
            #14*14*128 -> 28*28*1
            h4 = tf.layers.conv2d_transpose(h3, filters=cannel_size, kernel_size = 3,
                                            strides=2, padding='same')
            outputs = tf.tanh(h4)
            
        return outputs
    
    def discriminator(self, img_input, c, reuse=False, alpha=0.01):
        with tf.variable_scope('discriminator', reuse=reuse):
            
            # concat img_input and c
            # batch_size*28*28*1 + batch_size*10 -> batch_size*28*28*11
            c_shape = c.get_shape()
            c = tf.reshape(c, shape=(-1,1,1,c_shape[1]))
            
            img_shape = img_input.get_shape()
            c_shape = c.get_shape()
            
            img_input = tf.concat([img_input, c*tf.ones([self.batch_size,img_shape[1],img_shape[2],c_shape[3]])],
                                   axis=3)
            
            
            # 28*28*11 -> 14*14*128
            h1 = tf.layers.conv2d(img_input, filters=128, kernel_size=3, 
                           strides=2, padding='same')
            h1 = tf.maximum(alpha*h1, h1)
            h1 = tf.nn.dropout(h1, keep_prob=0.8)
            
            #14*14*128 -> 7*7*256
            h2 = tf.layers.conv2d(h1, filters=256, kernel_size=3,
                           strides=2, padding='same')
            h2 = tf.layers.batch_normalization(h2,training=True)
            h2 = tf.maximum(alpha*h2, h2)
            h2 = tf.nn.dropout(h2, keep_prob = 0.8)
            
            #7*7*256 -> 4*4*512
            h3 = tf.layers.conv2d(h2, filters=512, kernel_size=3,
                           strides=2, padding='same')
            h3 = tf.layers.batch_normalization(h3,training=True)
            h3 = tf.maximum(alpha*h3, h3)
            h3 = tf.nn.dropout(h3, keep_prob = 0.8)            
            
            #4*4*512 -> 1
            h4 = tf.reshape(h3, (-1,4*4*512))
            outputs = tf.layers.dense(h4, 1)
            #outputs = tf.sigmoid(logits)            
            
        return outputs
    
    def train(self, learning_rate=0.001):
        #model loss
        real_out = self.discriminator(self.x, self.c) # prob of real input pair
        noise_out = self.discriminator(self.generator(self.noise, self.c, is_train=True), 
                                       self.c, reuse=True) # prob of noise input pair
        no_match_out = self.discriminator(self.random_x, self.c, reuse=True) # prob of no matched pair        
        # G losses
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=noise_out, 
                                                                        labels=tf.ones_like(noise_out)*0.9))        
        
        
        #D losses

        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_out,
                                                                             labels=tf.ones_like(real_out)*0.9))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=noise_out,
                                                                             labels=tf.zeros_like(noise_out)))
        
        d_loss_not_match = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=no_match_out,
                                                                             labels=tf.zeros_like(no_match_out)))
        
        D_loss = d_loss_real + d_loss_fake + d_loss_not_match
        
        #optimizers        
        g_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
        d_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        
        D_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.4).minimize(D_loss, var_list=d_vars)
        G_opt = tf.train.AdamOptimizer(learning_rate, beta1=0.4).minimize(G_loss, var_list=g_vars)
        
        
        return G_opt,D_opt,G_loss,D_loss

if __name__ == '__main__':    
    print('---DCGAN for mnist data---')
    #create GAN model
    z_size = 100  #noise size
    batch_size = 100 #batch size
    epoches = 1 # times of training
    
    #tarining set
    mnist = input_data.read_data_sets('.\mnist', one_hot=True)
    random_data = list(mnist.test.images)
    #random_input = np.array(random.sample(random_data, batch_size))
    
    #cgan model    
    gan = cgan(z_size,batch_size)
    G_opt,D_opt,G_loss,D_loss = gan.train() #return optimizer of G and d, also losses of G and D

    
    with tf.Session() as sess:
        
        saver = tf.train.Saver() # saver for saving trained model's parameters
        
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoches):
            for batch_i in range(int(mnist.train.num_examples/batch_size)):
                batch,b_y = mnist.train.next_batch(batch_size)
                b_x = np.reshape(batch,(batch_size, 28, 28, 1))
                b_x = b_x * 2 - 1 # scale to -1, 1
                #sample images from dataset for training
                random_input = np.array(random.sample(random_data, batch_size)).reshape((batch_size, 28, 28, 1)) 
                
                # train D
                batch_noise1 = np.random.uniform(-1, 1, size=(batch_size, z_size))
                d_,d_l = sess.run([D_opt,D_loss], feed_dict={gan.x : b_x,
                                                             gan.noise : batch_noise1,
                                                             gan.c : b_y,
                                                             gan.random_x : random_input})
                
                # then train G
                batch_noise2 = np.random.uniform(-1, 1, size=(batch_size, z_size))
                g_,g_l = sess.run([G_opt,G_loss], feed_dict={gan.noise:batch_noise2,
                                                             gan.c:b_y})
                
                print(d_l,'---',g_l)
        
        
        # save model
        saver.save(sess, './CGAN-mnist-model_epoches_)'+str(epoches))
        '''
        # load trained model
        saver = tf.train.import_meta_graph('GAN-mnist-model.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        '''
        #test
        test_c = np.array([np.zeros((1,10)) for i in range(10)]).reshape((10,10))
        for i in range(10):
            test_c[i][i] = 1
        batch_noise = np.random.uniform(-1, 1, size=(10, z_size))
        test_img = sess.run([gan.generator(gan.noise, gan.c, is_train = False)], 
                             feed_dict={gan.noise:batch_noise, gan.c:test_c})
        test_img = np.array(test_img)
        
        #show images
        fig=plt.figure(figsize=(8,8))
        for i in range(10):
            img = test_img[0][i].reshape([28, 28])
            ax=fig.add_subplot(5,2,i+1) #3*3 figure ith position
            ax.imshow(img,cmap='Greys_r')
        plt.show() # show test output with auto-generated 9 images
