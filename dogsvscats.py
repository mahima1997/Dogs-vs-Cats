
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 14:00:34 2017

@author: mah
"""

#using k-NN classifier for image classification

# import the necessary packages
import numpy as np
import numpy
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from imutils import paths
import numpy as np
import argparse       #this is for writing user friendly cmd line interfaces
import imutils        #package for resizing, rotating image, etdc
import cv2
import os


train_data=pd.read_csv("dogsvscatstrain.csv")
le = LabelEncoder()
train_data['labels'] = le.fit_transform(train_data['labels'])
print(train_data.head)
array=train_data.values
rawImages = array[:,2]
features = array[:,1]
labels=array[:,0]
#rawImages=np.array(train_data['rawImages'])
#features=np.array(train_data['features'])
#labels=np.array(train_data['labels'])
print(rawImages.shape)
print(features)
print(labels)
# partition the data into training and testing splits, using 75%
# of the data for training and the remaining 25% for testing
(trainRI, testRI, trainRL, testRL) = train_test_split(
	rawImages, labels, test_size=0.25, random_state=42)
(trainFeat, testFeat, trainLabels, testLabels) = train_test_split(
	features, labels, test_size=0.25, random_state=42)
print(trainRI)

# train and evaluate a k-NN classifer on the raw pixel intensities
print("[INFO] evaluating raw pixel accuracy...")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
#------------WHY ERROR Can't convert string to float--------#
model.fit(trainRI, trainRL)
acc = model.score(testRI, testRL)
print("[INFO] raw pixel accuracy: {:.2f}%".format(acc * 100))

# train and evaluate a k-NN classifer on the histogram
# representations
print("[INFO] evaluating histogram accuracy...")
model = KNeighborsClassifier(n_neighbors=1,n_jobs=-1)
model.fit(trainFeat, trainLabels)
acc = model.score(testFeat, testLabels)
print("[INFO] histogram accuracy: {:.2f}%".format(acc * 100))

#Calculating results for the test dataset using histogram features
aux = pd.read_csv('dogsvscatstest.csv')
output=model.predict(aux['features'])
df_output = pd.DataFrame()
df_output['labels'] = aux['labels']
df_output['prediction'] = output
df_output[['labels','prediction']].to_csv('dogsvscatsKNNsubmission.csv',index=False)