
from __future__ import print_function
import tensorflow as tf
from mvpa2.suite import *
import numpy as np
import os 

dir_path = os.path.dirname(os.path.realpath(__file__))
subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)
ds = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
                  targets=attrs.labels,
                  chunks=attrs.chunks)
#                  mask=os.path.join(subjpath, 'mask4_vt.nii.gz'))
# preprocessing
poly_detrend(ds, polyord=1, chunks_attr='chunks')
zscore(ds, param_est=('targets', ['rest']), dtype='float32')

# delete rest samples
interesting = np.array([i in ['scissors', 'face', 'cat', 'shoe', 'house', 'scrambledpix', 'bottle', 'chair'] for i in ds.sa.targets])
ds = ds[interesting]

# extracts the label vector and assigns integers instead of class labels
ds.targets[ds.targets=='scissors'] = 0
ds.targets[ds.targets=='face'] = 1
ds.targets[ds.targets=='cat'] = 2
ds.targets[ds.targets=='shoe'] = 3
ds.targets[ds.targets=='house'] = 4
ds.targets[ds.targets=='scrambledpix'] = 5
ds.targets[ds.targets=='bottle'] = 6
ds.targets[ds.targets=='chair'] = 7
ds.targets = ds.targets.astype(int)



# partition the dataset
# divides the 12 chucks into 6 and 6 subsets
ds_train = ds[ds.chunks < 6]
ds_test = ds[ds.chunks >= 6]
del ds


# Network Parameters
n_input = 164025  
#n_input = 163840 # data input 
n_classes = 8 # fMRI total number of classes
dropout = 0.75 # Dropout, probability to keep units

# getting the data into shape
train = np.append(np.ones((ds_train.samples.shape[0], n_input - 		ds_train.samples.shape[1])), ds_train.samples, axis=1)
#train = ds_train.samples
targets = ds_train.targets
labels = np.zeros((ds_train.samples.shape[0], n_classes))
for i in range(0, ds_train.samples.shape[0]):
	labels[i, targets[i]] = 1

del ds_train
test = np.append(np.ones((ds_test.samples.shape[0], n_input - 	   			ds_test.samples.shape[1])), ds_test.samples, axis=1)
#test = ds_test.samples
test_targets = ds_test.targets
test_labels = np.zeros((ds_test.samples.shape[0], n_classes))
for i in range(0, ds_test.samples.shape[0]):
	test_labels[i, test_targets[i]] = 1
del ds_test


# Parameters
learning_rate = 0.01
training_iters = 10000
batch_size = 300
display_step = 1


# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)


# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 405, 405, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32])),
    'bc2': tf.Variable(tf.random_normal([64])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()



# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Shuffle the data
        perm = np.arange(train.shape[0])
        np.random.shuffle(perm)
        train = train[perm]
        labels = labels[perm]
        # select next batch
        batch_x = train[0:batch_size]
        batch_y = labels[0:batch_size]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    # for saving the data to tensorboard
    summary_writer = tf.summary.FileWriter(dir_path, graph=sess.graph)

    # Calculate accuracy for the test data
    print("Testing Accuracy:", \
	sess.run(accuracy, feed_dict={x: test,
			              y: test_labels,
			              keep_prob: 1.}))
