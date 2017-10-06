# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:36:46 2017

@author: mah
"""
#HDF5 is a data model, library, and file format for storing and managing data. It supports an unlimited variety of datatypes,
#and is designed for flexible and efficient I/O and for high volume and complex data (more info).
#TFLearn can directly use HDF5 formatted data.


#Using CNN classification

#future is the missing compatibility layer between Python 2 and Python 3.
#It allows to use a single, clean Python 3.x-compatible codebase to support
#both Python 2 and Python 3 with minimal overhead(burden cost i.e resource consumed or lost in completing the process).
from __future__ import division, print_function, absolute_import    # _future_ is a real module
from skimage import color, io
from scipy.misc import imresize
import numpy as np
from sklearn.cross_validation import train_test_split
import os
from glob import glob   #glob module has two functions that either return a list or an 
                        #iterator of files in a directory using shell pattern matching. 
                        #for example, to return a list of all jpg files in a directory
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn.metrics import Accuracy

###Import picture files
files_path = 'train/'

cat_files_path = os.path.join(files_path, 'cat*.jpg')  #Asterisk is a wildcard character. It means match on everything. 
dog_files_path = os.path.join(files_path, 'dog*.jpg')  #So cat* means match on all files starting with cat in the directory.

cat_files = sorted(glob(cat_files_path))  #Return a possibly-empty list of path names that match pathname, which must be a 
dog_files = sorted(glob(dog_files_path))  #string containing a path specification. 

n_files = len(cat_files) + len(dog_files)   #len bcoz cat_files is a list returned by glob
print(n_files)

size_image = 64

allX = np.zeros((n_files, size_image, size_image, 3), dtype='float64')
ally = np.zeros(n_files)
count = 0
for f in cat_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 0
        count += 1
    except:
        continue

for f in dog_files:
    try:
        img = io.imread(f)
        new_img = imresize(img, (size_image, size_image, 3))
        allX[count] = np.array(new_img)
        ally[count] = 1
        count += 1
    except:
        continue
    
###################################
# Prepare train & test samples
################################### 

X, X_test, Y, Y_test = train_test_split(allX, ally, test_size=0.1, random_state=42)
Y = to_categorical(Y, 2)
Y_test = to_categorical(Y_test, 2)

###################################
# Image transformations
################################### 
#TFLearn data stream is designed with computing pipelines in order to speed-up training (by pre-processing data on 
#CPU while GPU(Graphics processing unit) is performing model training).

####normalisation of images####
#In image processing, normalization is a process that changes the range of pixel intensity 
#values. Applications include photographs with poor contrast due to glare, for example. 
#Normalization is sometimes called contrast stretching or histogram stretching.
#It is usually to bring the image, or other type of signal, into a range that is 
#more familiar or normal to the senses, hence the term normalization. 

img_prep = ImagePreprocessing()           #no arguments passed in the function implies 
img_prep.add_featurewise_zero_center()    #preprocessing and augmentation will be done 
img_prep.add_featurewise_stdnorm()	     #on the training data by default.  

# Create extra synthetic training data by flipping & rotating images
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

###################################
# Define network architecture
###################################

#Adding the above methods into an 'input_data' layer
#Input is a 32x32 image with 3 color channels (red, green and blue)
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)

# 1: Convolution layer with 32 filters, each 3x3x3
conv_1 = conv_2d(network, 32, 3, activation='relu', name='conv_1') #default name is Conv2D

# 2: Max pooling layer
network = max_pool_2d(conv_1, 2) #2 is the pooling kernel size #default name for this layer is 'MAXPOOL2D'

# 3: Convolution layer with 64 filters
conv_2 = conv_2d(network, 64, 3, activation='relu', name='conv_2') #ReLu is rectified linear 										#unit

# 4: Convolution layer with 64 filters
conv_3 = conv_2d(conv_2, 64, 3, activation='relu', name='conv_3') #64 is number of filters
								  #3 is size of filters
# 5: Max pooling layer
network = max_pool_2d(conv_3, 2)

# 6: Fully-connected 512 node layer
network = fully_connected(network, 512, activation='relu')

# 7: Dropout layer to combat overfitting
network = dropout(network, 0.5)

# 8: Fully-connected layer with two outputs
network = fully_connected(network, 2, activation='softmax')

# Configure how the network will be trained
acc = Accuracy(name="Accuracy")

#The regression layer is used in TFLearn to apply a regression (linear or logistic) to the provided input.
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy', #this loss type is used only when the target variable is categorical
                     learning_rate=0.0005, metric=acc)
#There are many ways to define a learning rate. It can be a function, or just a fixed number. 
#The 'learning rate' is how quickly a network abandons old beliefs for new ones. 
#In general, you want to find a learning rate that is low enough that the network converges to something useful, but high enough 
#that we don't have to spend years training it.

#If a child sees 10 examples of cats and all of them have orange fur, it will think that cats have orange fur and will look for orange
#fur when trying to identify a cat. Now it sees a black a cat and her parents tell her it's a cat (supervised learning). With a large “learning rate”, 
#it will quickly realize that “orange fur” is not the most important feature of cats. With a small learning rate, it will think that 
#this black cat is an outlier and cats are still orange.

# Wrap the network in a model object
model = tflearn.DNN(network, checkpoint_path='model_cat_dog_6.tflearn', max_checkpoints = 3,
tensorboard_verbose = 3, tensorboard_dir='tmp/tflearn_logs/')	#DNN is Deep Neural Network

#checkpoint_path: (str) -> Path to store model checkpoints. If None, no model checkpoint will be saved. Default: None.
#Checkpoint is an approach where a snapshot of the state of the system is taken in case of system failure. If there is a problem,
#not all is lost. The checkpoint may be used directly, or used as the starting point for a new run, picking up where it left off.

###################################
# Train model for 100 epochs
###################################

#epoch can be defined as one forward pass and one backward pass of all training data while iteration is one forward pass and one
#backward pass of each batch size. large training data examples can not fit in memory so we process the data by dividing in batch sizes. 

# one pass = one forward pass + one backward pass
#Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.

#Training can go on for 100s of epochs.

model.fit(X, Y, validation_set=(X_test, Y_test), batch_size=500,
      n_epoch=100, run_id='model_cat_dog_6', show_metric=True)

#show_metric: (bool)-> Display or not accuracy at every step.
#run_id: (str)-> Give a name for this run. (Useful for Tensorboard).

#Given that deep learning models can take hours, days and even weeks to train, it is important to save and load them from disk
model.save('model_cat_dog_6_final.tflearn')
print("done")

# To load this model for further predictions
#model.load('model_cat_dog_6_final.tflearn')






















