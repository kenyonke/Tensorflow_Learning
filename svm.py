# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#data preprocessing
iris = datasets.load_iris()
X = np.array([[x[0], x[3]] for x in iris.data]) #select 2-dimention features for visualization 
Y = np.array([[1.] if y == 0 else [-1.] for y in iris.target])

train_indices = np.random.choice(len(X), round(len(X)*0.8),replace=False)
test_indices = np.array(list(set(range(len(X))) - set(train_indices)))

train_x = X[train_indices]
train_y = Y[train_indices]

test_x = X[test_indices]
test_y = Y[test_indices]

data_x = tf.placeholder(tf.float32,[None,2])
data_y = tf.placeholder(tf.float32,[None,1])

#SVM model
W = tf.Variable(tf.zeros(shape=[2,1]), dtype=tf.float32, name = 'W')
B = tf.Variable(tf.zeros(shape=[1,1]), dtype=tf.float32, name = 'B')
pred = tf.matmul(data_x,W) + B

#Hinge Loss
hinge_loss = tf.maximum(0., 1 - data_y*pred)

#L2 normalization 
l2_norm = tf.reduce_sum(tf.square(W))

#Loss function
loss = tf.reduce_mean(hinge_loss + 0.02 * l2_norm)

#Gradient desecent optimizer
opt = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

#Accuracy function
def accuracy(predictions,labels):
    if len(predictions) != len(labels):
        raise('the number of predictions and labels are different!!!')
        return 0
    else:
        cor = 0
        sum_counts = len(predictions)
        for i in range(sum_counts):
            if predictions[i]<0 and labels[i]==-1:
               cor += 1
            elif predictions[i]>0 and labels[i]==1:
                cor += 1
        return cor/sum_counts

#training and testing    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())   
    
    for i in range(2000):
        loss_val,_ = sess.run([loss,opt],feed_dict = {data_x:train_x,data_y:train_y})
        #print('loss value:',loss_val)

    predictions = sess.run([pred],feed_dict = {data_x:test_x})
    print('Test accuracy: ',accuracy(predictions[0],test_y))
    
    #Class1 and Class2
    c1_x1 = [features[1] for i, features in enumerate(X) if Y[i][0] == 1.]
    c1_x2 = [features[0] for i, features in enumerate(X) if Y[i][0] == 1.]
    c2_x1 = [features[1] for i, features in enumerate(X) if Y[i][0] == -1.]
    c2_x2 = [features[0] for i, features in enumerate(X) if Y[i][0] == -1.]
    
    #Linear Separator
    x1_vals = [d[1] for d in X]
    [[w1], [w2]] = sess.run(W)
    [[b]] = sess.run(B)
    x2_vals = [(-(w2*x1 + b)/w1) for x1 in x1_vals]
    
    #plot the svm and data
    plt.plot(c1_x1, c1_x2, 'o', label='Class 1')
    plt.plot(c2_x1, c2_x2, 'x', label='Class -1')
    plt.plot(x1_vals, x2_vals, 'r-',linewidth=2)
    plt.legend(loc='lower right')
    plt.title('SVM')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.ylim(-10,10)
    plt.show()