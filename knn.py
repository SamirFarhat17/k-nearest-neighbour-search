import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import heapq
from scipy.spatial import distance
from tqdm.notebook import tqdm, trange
import copy
import heapq
from scipy import spatial
import os

# function for unpickling the CIFAR10 dataset
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# import CIFAR10 image batches
DATA_ROOT = 'CIFAR10/cifar-10-batches-py/'
batch1 = unpickle(DATA_ROOT + "data_batch_1")
batch2 = unpickle(DATA_ROOT + "data_batch_2")
batch3 = unpickle(DATA_ROOT + "data_batch_3")
batch4 = unpickle(DATA_ROOT + "data_batch_4")
batch5 = unpickle(DATA_ROOT + "data_batch_5")
test_batch = unpickle(DATA_ROOT + "test_batch")

images = np.concatenate([batch1[b'data'], batch2[b'data'], batch3[b'data'], batch4[b'data'], batch5[b'data'], test_batch[b'data']])

# confirm correct size of data array
print("Data dimensions: ", np.shape(images))

## functions for displaying cifar images
def cifar10_plot_single(data, im_idx=0):
    im = data[im_idx, :]
    
    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))

    print("shape: ", img.shape)        
    
    plt.imshow(img) 
    plt.pause(0.3)

def cifar10_plot_examples(data):
  
  for i in range(9):
    im = data[i, :]
    im_r = im[0:1024].reshape(32, 32)
    im_g = im[1024:2048].reshape(32, 32)
    im_b = im[2048:].reshape(32, 32)

    img = np.dstack((im_r, im_g, im_b))
    # define subplot
    plt.subplot(330 + 1 + i)
    # plot 
    plt.imshow(img)
    plt.pause(0.3)
'''
cifar10_plot_examples(images)
cifar10_plot_single(images, im_idx=4)
'''
def pca_reduce(images, n_components):
  pca = PCA(n_components = n_components)
  images_reduced = pca.fit_transform(images)
  images_recovered = pca.inverse_transform(images_reduced)
  
  print("Image data reduced from: {} to {}".format(images.shape, images_reduced.shape))
  print()
  print("Explained variance ratio of each component: ")
  print(pca.explained_variance_ratio_)
  
  return images_reduced, images_recovered

# to inspect the PCA predicted images
def plot_pca_image(images, images_recovered, idx = 1):
    cifar10_plot_single(images, im_idx= idx)
    cifar10_plot_single(images_recovered.astype(int), im_idx = idx)

images_reduced, images_recovered = pca_reduce(images, 5)
'''
print("\n----------------------\n" + str(images_recovered))
print("\n----------------------\n" + str(images_reduced))
'''

plot_pca_image(images, images_recovered, idx = 4)