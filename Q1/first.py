import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import cv2                          # for image loading and colour conversion
import tensorflow as tf             # for bulk image resize
import os
import glob
import random

from sklearn import decomposition
from sklearn import discriminant_analysis
from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import Utility_PCA as ute
import Utility2 as ute2

#load the data
data_dir = "./Data"
train_X, train_Y, gallery_X, gallery_Y, probe_X, probe_Y = ute.load_data(data_dir)
print(train_X.shape)
print(train_Y.shape)
print(gallery_X.shape)
print(gallery_Y.shape)
print(probe_X.shape)
print(probe_Y.shape)



#plot the first 50 original images
ute.plot_images(gallery_X, gallery_Y)

# # create a pair and display a pair
# pair_test = ute.pair_generator(train_X, train_Y, 10)
# x, y = next(pair_test)
# ute.plot_pairs(x, y)

# # create a triplet and display a triplet
# triplet_test = ute.triplet_generator(train_X, train_Y, 9)
# x, _ = next(triplet_test)
# ute.plot_triplets(x)

# resize data 
train_X_small = ute.resize(train_X, (64, 32))
gallery_X_small = ute.resize(gallery_X, (64, 32))
probe_X_small = ute.resize(probe_X, (64, 32))


# and convert to grayscale
# train_X_gray = ute.convert_to_grayscale(train_X)
# gallery_X_gray = ute.convert_to_grayscale(gallery_X)
# probe_X_gray = ute.convert_to_grayscale(probe_X)
train_X_small_gray = ute.convert_to_grayscale(train_X_small)
gallery_X_small_gray = ute.convert_to_grayscale(gallery_X_small)
probe_X_small_gray = ute.convert_to_grayscale(probe_X_small)

#vectorise
train_feat = ute.vectorise(train_X)
gallery_feat = ute.vectorise(gallery_X)
probe_feat = ute.vectorise(probe_X)

#ute2.MeanImg(train_feat) #the plotting of the mean face requires the face be of a smaller specification

print()
print("New Shapes")
print()
print(np.shape(train_X))
print(np.shape(train_X_small))
print(np.shape(train_X_small_gray))

print(np.shape(gallery_X))
print(np.shape(gallery_X_small))
print(np.shape(gallery_X_small_gray))

print(np.shape(probe_X))
print(np.shape(probe_X_small))
print(np.shape(probe_X_small_gray))


# pca = decomposition.PCA()
# pca.fit(train_feat)
# transformed = pca.transform(train_X)

#Visualise the principal components - non vectorised
#ute2.VisualisePC(pca, 30) #fails aparently trying to shape 24576 into 32x32

# plot some resized and grayscale images
ute.plot_images(gallery_X_small_gray, gallery_Y)




plt.show()