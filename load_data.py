####################################                 
####    Loading Haxby dataset   ####
####        subject 1        ####
####################################

#### Version    : 1.0
#### Date       : 20 May 2017

####### Import #########
from mvpa2.suite import *
import numpy as np
import os 

 

def LoadHaxby():
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
	
	data = ds.samples
	data_label = ds.targets

	return data, data_label
