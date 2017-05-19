from mvpa2.suite import *
import numpy as np

# prepares the fmri_dataset prameters
subjpath = os.path.join('/home/arash/Desktop/Dropbox/2017-Spring/CS464/Dataset/subj6')
attrs = SampleAttributes(os.path.join(subjpath, 'labels.txt'),header=True)

# loads the fmri file into and ds object with all the attributes
ds = fmri_dataset(samples=os.path.join(subjpath,'bold.nii.gz'),
targets=attrs.labels, chunks=attrs.chunks)

# extracts the data matrix as integers
data = ds.samples.astype(int)

# extracts the label vector and assigns integers instead of class labels
labels = ds.targets
labels[labels=='rest'] = 0
labels[labels=='scissors'] = 1
labels[labels=='face'] = 2
labels[labels=='cat'] = 3
labels[labels=='shoe'] = 4
labels[labels=='house'] = 5
labels[labels=='scrambledpix'] = 6
labels[labels=='bottle'] = 7
labels[labels=='chair'] = 8
labels = labels.astype(int)

print data.shape

# saves the files
np.savetxt("./data/subj6/unmasked/data.txt", data, fmt='%d')
np.savetxt("./data/subj6/unmasked/labels.txt", labels, fmt='%d')

#print ds.summary()

#
# # Code for the heatmap of the dataset for "smelling"
# ##############################################################333
# # next only works with floating point data
# ds.samples = ds.samples.astype('float')
#
# # look at sample similarity
# # Note, the decreasing similarity with increasing temporal distance
# # of the samples
# pl.figure(figsize=(14, 6))
# pl.subplot(121)
# plot_samples_distance(ds)
# pl.title('Sample distances (sorted by chunks)')
# plot_samples_distance(ds, sortbyattr='chunks')
#
# # similar distance plot, but now samples sorted by their
# # respective targets, i.e. samples with same targets are plotted
# # in adjacent columns/rows.
# # Note, that the first and largest group corresponds to the
# # 'rest' condition in the dataset
# pl.subplot(122)
# plot_samples_distance(ds, sortbyattr='targets')
# pl.title('Sample distances (sorted by targets)')
#
# if cfg.getboolean('examples', 'interactive', True):
#     pl.show()
