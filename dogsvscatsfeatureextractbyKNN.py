# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 18:22:22 2017

@author: mah
"""

# import the necessary packages

import numpy as np
import numpy
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse       #this is for writing user friendly cmd line interfaces
import imutils        #package for resizing, rotating image, etdc
import cv2
import os

def image_to_feature_vector(image, size=(32, 32)):
	# resize the image to a fixed size, then flatten the image into
	# a list of raw pixel intensities
	return cv2.resize(image, size).flatten()      #flatten returns vector

def extract_color_histogram(image, bins=(8, 8, 8)):
	# extract a 3D color histogram from the HSV color space using
	# the supplied number of `bins` per channel
	hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	hist = cv2.calcHist([hsv], [0, 1, 2], None, bins,
		[0, 180, 0, 256, 0, 256])
	# handle normalizing the histogram in OpenCV 3
	cv2.normalize(hist, hist)
	# return the flattened histogram as the feature vector
	return hist.flatten()      
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d","--dataset", required=True, help="path to input dataset")    to give input of datset through cmd argument
ap.add_argument("-d","--dataset",default="train", help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#We are now ready to prepare our images for feature extraction:
#grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

#Now lets extract features from our dataset
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (as our
	# path has the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
 
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
 
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
 
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

#------------dump this data into a file with columns- 'labels' 'features' and 'rawimages'-------------
df=pd.DataFrame({'labels':labels,'features':features,'rawImages':rawImages})
df[['labels','features','rawImages']].to_csv('dogsvscatstrain.csv',index=False)

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))
 
 #---same as above for test data--
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
#ap.add_argument("-d","--dataset", required=True, help="path to input dataset")    to give input of datset through cmd argument
ap.add_argument("-d","--dataset",default="test", help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1,
	help="# of nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1,
	help="# of jobs for k-NN distance (-1 uses all available cores)")
args = vars(ap.parse_args())

#We are now ready to prepare our images for feature extraction:
#grab the list of images that we'll be describing
print("[INFO] describing images...")
imagePaths = list(paths.list_images(args["dataset"]))
 
# initialize the raw pixel intensities matrix, the features matrix,
# and labels list
rawImages = []
features = []
labels = []

#Now lets extract features from our dataset
# loop over the input images
for (i, imagePath) in enumerate(imagePaths):
	# load the image and extract the class label (as our
	# path has the format: /path/to/dataset/{class}.{image_num}.jpg
	image = cv2.imread(imagePath)
	label = imagePath.split(os.path.sep)[-1].split(".")[0]
 
	# extract raw pixel intensity "features", followed by a color
	# histogram to characterize the color distribution of the pixels
	# in the image
	pixels = image_to_feature_vector(image)
	hist = extract_color_histogram(image)
 
	# update the raw images, features, and labels matricies,
	# respectively
	rawImages.append(pixels)
	features.append(hist)
	labels.append(label)
 
	# show an update every 1,000 images
	if i > 0 and i % 1000 == 0:
		print("[INFO] processed {}/{}".format(i, len(imagePaths)))

#------------dump this data into a file with columns- 'labels' 'features' and 'rawimages'-------------
df=pd.DataFrame({'labels':labels,'features':features,'rawImages':rawImages})
df[['labels','features','rawImages']].to_csv('dogsvscatstest.csv',index=False)

# show some information on the memory consumed by the raw images
# matrix and features matrix
rawImages = np.array(rawImages)
features = np.array(features)
labels = np.array(labels)
print("[INFO] pixels matrix: {:.2f}MB".format(
	rawImages.nbytes / (1024 * 1000.0)))
print("[INFO] features matrix: {:.2f}MB".format(
	features.nbytes / (1024 * 1000.0)))