
import sklearn
from pathlib import Path
import os
import glob
import shutil
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

#####################################
# Defining a skeleton for our       #
# DataFrame                         #
#####################################

sk_struct = {
    'photo_name' : [],
    'flattenPhoto' : []
    }
#######################################################################################
#      The Approach is to apply transfer learning hence using Resnet50 as my          #
#      pretrained model                                                               #
#######################################################################################

"""
some notes- wioll experiment with dense nets also- residual nets from the application module seem to lackdepth while looking at the image
so will experment with the other too
"""



MyModel = tf.keras.models.Sequential()
MyModel.add(tf.keras.applications.DenseNet201(
    include_top = False, weights='imagenet', pooling='avg')
    )


# freezing weights for 1st layer
MyModel.layers[0].trainable = False

#MyModel.add(tf.keras.layers.Flatten())

#MyModel.add(tf.keras.layers.Dense(66, activation='relu'))

MyModel.add(tf.keras.layers.Dense(66, activation='softmax'))

#MyModel.add(tf.keras.layers.Dense(66, activation='relu'))
#MyModel.compile(optimizer=tf.keras.optimizers.Adam())

MyModel.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])




### Now defining dataloading Function
def LoadDataAndDoEssentials(path, h, w):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (h, w))
    
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)
    
    img = tf.keras.applications.densenet.preprocess_input(img)
    
    extractedFeatures = MyModel.predict(img)
    extractedFeatures = np.array(extractedFeatures)
    sk_struct['flattenPhoto'].append(extractedFeatures.flatten())

    
### with this all done lets write the iterrrative loop
def ReadAndStoreMyImages(path):

    filenames_only = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    for filename in filenames_only:
        sk_struct['photo_name'].append(filename)
        imagePath=path/filename
        filenames.append(imagePath)
        LoadDataAndDoEssentials(imagePath, 224, 224)
    



### lets give the address of our Parent directory and start
CURRENT_PATH_CWD = Path.cwd()
TRAINING_DATA_DIR = CURRENT_PATH_CWD/'DATA'
data_path=TRAINING_DATA_DIR
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAIN'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAINING_DATA'

filenames=[]

ReadAndStoreMyImages(TRAINING_DATA_DIR)


######################################################
#        lets now do clustering                      #
######################################################

X = np.array(sk_struct['flattenPhoto'], dtype = 'float64')



# #####################################
# Compute the KNN distance
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)
distances, indices = knn.kneighbors(X)
array_dist=knn.kneighbors_graph(X).toarray()
distances = np.sort(distances, axis=0)
distances = distances[:,1]

eps=distances.mean()

# Apply DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=5)
dbscan.fit(X)


labels = dbscan.labels_


# Count the number of clusters
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Number of clusters:', num_clusters)

for label in labels:
    new_path=data_path/str(label)
    if new_path not in data_path.glob("*"):
        os.mkdir(new_path)


# copies images in coresponding folder
for filename, label in zip(filenames, labels):
    path=data_path/str(label)
    shutil.copy(filename,path)

