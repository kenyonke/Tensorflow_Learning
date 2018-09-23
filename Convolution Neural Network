import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

BATCH_SIZE = 50
LR = 0.001              # learning rate

#load data
mnist = input_data.read_data_sets('.\mnist', one_hot=True)  # they has been normalized to range (0,1)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

'''
# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)   # (55000, 10)
plt.imshow(mnist.train.images[1].reshape((28, 28)), cmap='Wistia') #'cmap' is background color
plt.title(np.argmax(mnist.train.labels[1]))
plt.show()
'''
#create CNN 
tf_x = tf.placeholder(tf.float32, [None, 28*28]) / 255.0
image = tf.reshape(tf_x, [-1, 28, 28, 1])              # (batch, height, width, channel)

tf_y = tf.placeholder(tf.int32, [None, 10])            # input y

# CNN layers
conv1 = tf.layers.conv2d(   # shape (28, 28, 1)
    inputs=image,
    filters=16, #The number of filters is 16!!!
    kernel_size=5,
    strides=1,
    padding='same',
    activation=tf.nn.relu
)           # -> (28, 28, 16)
pool1 = tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2
)           # -> (14, 14, 16)
conv2 = tf.layers.conv2d(pool1, 32, 5, 1, 'same', activation=tf.nn.relu)    # -> (14, 14, 32)
pool2 = tf.layers.max_pooling2d(conv2, 2, 2)    # -> (7, 7, 32)
cnn_output = tf.reshape(pool2, [-1, 7*7*32])          # -> (7*7*32, )

softmax_w = tf.Variable(tf.zeros([7*7*32,10]))
softmax_b = tf.Variable(tf.zeros([10]))
output = tf.nn.softmax(tf.matmul(cnn_output,softmax_w) + softmax_b)
'''
output = tf.layers.dense(cnn_output, 10)              # output layer
'''
loss = tf.losses.softmax_cross_entropy(tf_y, logits=output)           # compute cost
training = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init)     # initialize var in graph

'''
# following function (plot_with_labels) is for visualization, can be ignored if not interested
from matplotlib import cm
from sklearn.manifold import TSNE

def plot_with_labels(lowDWeights, labels):
    plt.cla()
    X = lowDWeights[:, 0]
    Y = lowDWeights[:, 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer');
    plt.show()
    
plt.ion()
'''
for i in range(200):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, loss_ = sess.run([training, loss], feed_dict = {tf_x: b_x, tf_y: b_y})
    '''
    if i % 50 == 0:
        accuracy_, flat_representation = sess.run([accuracy, cnn_output], {tf_x: test_x, tf_y: test_y})
        print('Step:', i, '| train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

        # Visualization of trained flatten layer (T-SNE)
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000); plot_only = 500
        low_dim_embs = tsne.fit_transform(flat_representation[:plot_only, :])
        labels = np.argmax(test_y, axis=1)[:plot_only]
        plot_with_labels(low_dim_embs, labels)
plt.ioff()
'''

test_output = sess.run(output, {tf_x: test_x[:30]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:30], 1), 'real number')

sess.close()
