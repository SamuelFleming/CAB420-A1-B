import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt     # for plotting
import numpy as np                  # for reshaping, array manipulation
import cv2                          # for image loading and colour conversion
import tensorflow as tf             # for bulk image resize
import os
import glob
import random

import Utility_PCA as ute


def MeanImg(train_feat):
    #display mean person image
    meanperson = np.reshape(np.mean(train_feat, axis=0), (-1, 1));
    fig = plt.figure(figsize=[5, 5])
    ax = fig.add_subplot(1, 1, 1)
    meanperson_im = np.reshape(meanperson, (32, 32))    
    ax.imshow(meanperson_im.transpose(), cmap=plt.get_cmap('gray'))
    ax.set_axis_off()
    ax.set_title('A Very Average Face');


def VisualisePC(pca, num):
    fig = plt.figure(figsize=[20, 16])
    for i in range(num):
        ax = fig.add_subplot(8, 10, i + 1)
        pc = np.reshape(pca.components_[i,:], (32, 32))        
        ax.imshow(pc.transpose(), cmap=plt.get_cmap('gray'))
        ax.set_axis_off()