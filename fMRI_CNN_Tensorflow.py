# A simple CNN to predict certain characteristics of the human subject from MRI images.
# 3d convolution is used in each layer.
# Reference: https://www.tensorflow.org/get_started/mnist/pros, http://blog.naver.com/kjpark79/220783765651
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc

from __future__ import print_function
import tensorflow as tf
from mvpa2.suite import *
import numpy as np
import os 


width = 40
height = 64
depth = 64
nLabel = 8


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

# create one hot encoding of train and test labels
train_labels_onehot = np.zeros((ds_train.samples.shape[0], nLabel))
for i in range(0, ds_train.samples.shape[0]):
	train_labels_onehot[i, ds_train.targets[i]] = 1

test_labels_onehot = np.zeros((ds_test.samples.shape[0], nLabel))
for i in range(0, ds_test.samples.shape[0]):
	test_labels_onehot[i, ds_test.targets[i]] = 1








# Start TensorFlow InteractiveSession
import tensorflow as tf
sess = tf.InteractiveSession()

# Placeholders (MNIST image:28x28pixels=784, label=10)
x = tf.placeholder(tf.float32, shape=[None, width*height*depth]) # [None, 28*28]
y_ = tf.placeholder(tf.float32, shape=[None, nLabel])  # [None, 10]

## Weight Initialization
# Create lots of weights and biases & Initialize with a small positive number as we will use ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

## Convolution and Pooling
# Convolution here: stride=1, zero-padded -> output size = input size
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

# Pooling: max pooling over 2x2 blocks
def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 4, 4, 4, 1], strides=[1, 4, 4, 4, 1], padding='SAME')










## First Convolutional Layer
# Conv then Max-pooling. 1st layer will have 32 features for each 5x5 patch. (1 feature -> 32 features)
W_conv1 = weight_variable([5, 5, 5, 1, 32])  # shape of weight tensor = [5,5,1,32]
b_conv1 = bias_variable([32])  # bias vector for each output channel. = [32]

# Reshape 'x' to a 4D tensor (2nd dim=image width, 3rd dim=image height, 4th dim=nColorChannel)
x_image = tf.reshape(x, [-1,width,height,depth,1]) # [-1,28,28,1]
print(x_image.get_shape) # (?, 256, 256, 40, 1)  # -> output image: 28x28 x1

# x_image * weight tensor + bias -> apply ReLU -> apply max-pool
h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1) + b_conv1)  # conv2d, ReLU(x_image * weight + bias)
print(h_conv1.get_shape) # (?, 256, 256, 40, 32)  # -> output image: 28x28 x32
h_pool1 = max_pool_2x2(h_conv1)  # apply max-pool 
print(h_pool1.get_shape) # (?, 128, 128, 20, 32)  # -> output image: 14x14 x32


## Second Convolutional Layer
# Conv then Max-pooling. 2nd layer will have 64 features for each 5x5 patch. (32 features -> 64 features)
W_conv2 = weight_variable([5, 5, 5, 32, 64]) # [5, 5, 32, 64]
b_conv2 = bias_variable([64]) # [64]

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2) + b_conv2)  # conv2d, .ReLU(x_image * weight + bias)
print(h_conv2.get_shape) # (?, 128, 128, 20, 64)  # -> output image: 14x14 x64
h_pool2 = max_pool_2x2(h_conv2)  # apply max-pool 
print(h_pool2.get_shape) # (?, 64, 64, 10, 64)    # -> output image: 7x7 x64


## Densely Connected Layer (or fully-connected layer)
# fully-connected layer with 1024 neurons to process on the entire image

# *16 is removed to adjust to Haxby dataset which is 64*64*40 and not 256*256*40
W_fc1 = weight_variable([16*3*64, 1024]) 
b_fc1 = bias_variable([1024]) # [1024]]

# *16 is removed to adjust to Haxby dataset which is 64*64*40 and not 256*256*40
h_pool2_flat = tf.reshape(h_pool2, [-1, 16*3*64])
print(h_pool2_flat.get_shape)  # (?, 2621440)
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  # ReLU(h_pool2_flat x weight + bias)
print(h_fc1.get_shape) # (?, 1024)  # -> output: 1024

## Dropout (to reduce overfitting; useful when training very large neural network)
# We will turn on dropout during training & turn off during testing
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.get_shape)  # -> output: 1024

## Readout Layer
W_fc2 = weight_variable([1024, nLabel]) # [1024, 10]
b_fc2 = bias_variable([nLabel]) # [10]

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print(y_conv.get_shape)  # -> output: 10

## Train and Evaluate the Model
# set up for optimization (optimizer:ADAM)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())

batch_size = 10
# Include keep_prob in feed_dict to control dropout rate.
for i in range(100):
    # Shuffle the data
    perm = np.arange(ds_train.samples.shape[0])
    np.random.shuffle(perm)
    ds_train.samples = ds_train.samples[perm]
    train_labels_onehot = train_labels_onehot[perm]
    # select next batch
    batch_x = ds_train.samples[0:batch_size]
    batch_y = train_labels_onehot[0:batch_size]
    # Logging every 100th iteration in the training process.
    #if i%5 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch_x, y_: batch_y, keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch_x, y_: batch_y, keep_prob: 0.5})

# Evaulate our accuracy on the test data
print("test accuracy %g"%accuracy.eval(feed_dict={x: ds_test.samples, y_: test_labels_onehot, keep_prob: 1.0}))




