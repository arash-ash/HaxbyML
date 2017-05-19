from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


from mvpa2.suite import *
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj1')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)
ds = fmri_dataset(samples=os.path.join(subjpath, 'bold.nii.gz'),
                  targets=attrs.labels,
                  chunks=attrs.chunks,
                  mask=os.path.join(subjpath, 'mask4_vt.nii.gz'))
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




# Building convolutional network
n_input = 784
n_classes = 8

network = input_data(shape=[None, 28, 28, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, n_classes, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.01,
                     loss='categorical_crossentropy', name='target')



# getting data in shape

X = np.append(np.ones((ds_train.samples.shape[0], n_input - ds_train.samples.shape[1])), ds_train.samples, axis=1)
X = X.reshape([-1, 28, 28, 1])
targets = ds_train.targets
Y = np.zeros((ds_train.samples.shape[0], n_classes))
for i in range(0, ds_train.samples.shape[0]):
	Y[i, targets[i]] = 1

testX = np.append(np.ones((ds_test.samples.shape[0], n_input - 		   ds_test.samples.shape[1])), ds_test.samples, axis=1)
testX = testX.reshape([-1, 28, 28, 1])
test_targets = ds_test.targets
testY = np.zeros((ds_test.samples.shape[0], n_classes))
for i in range(0, ds_test.samples.shape[0]):
	testY[i, test_targets[i]] = 1


# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': X}, {'target': Y}, n_epoch=2000,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=10, show_metric=True, run_id='convnet_haxby')


