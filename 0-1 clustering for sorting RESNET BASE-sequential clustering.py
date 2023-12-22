
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
#      The Approach is to apply transfer learning                                     #
#######################################################################################



MyModel=tf.keras.applications.ResNet101(
    include_top = False, weights='imagenet', pooling='avg')

"""
Some topics about what happened here. A model has been definied in the Resnet structure from Keras
It is a sequential model and has a number of layers in it
At the moment it has only recieved the input from the resnet structure
"""
# freezing weights for 1st layer
#MyModel.layers[0].trainable = False
MyModel.trainable=False

MyModel.compile(optimizer=tf.keras.optimizers.Adam())

MyModel.trainable=True

MyModel.layers[-1].trainable = False
MyModel.compile(optimizer=tf.keras.optimizers.Adam())


"""
Here ends the model as defined by resnet
"""




### Now defining dataloading Function

def LoadDataAndDoEssentials(path, h, w):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (h, w))
    
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)

    img = tf.keras.applications.resnet50.preprocess_input(img)

        
    extractedFeatures = MyModel.predict(img)
    
    """
    print("AICI SUNT IMG DVS")
    print(type(img))
    print(img)
    print(img.shape)
    print("si aici xtracted features")
    print(type(extractedFeatures))
    print(extractedFeatures)
    print(extractedFeatures.shape)
    """
    
    extractedFeatures = np.array(extractedFeatures)

    #THIS IS THE IMPORTANT ROW IN THIS PART
    sk_struct['flattenPhoto'].append(extractedFeatures.flatten())


    
### with this all done lets write the iterrrative loop

def ReadAndStoreMyImages(path):

    filenames_only = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    i=0
    for filename in filenames_only:

        #THIS IS THE IMPORTANT ROW IN THIS PART
        sk_struct['photo_name'].append(filename)
        
        imagePath=path/filename
        filenames.append(imagePath)
        LoadDataAndDoEssentials(imagePath, 112, 112)
        '''
        if i>1: 
            break
        else: i=i+1
        '''
        
        
    



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

"""
Basically it gets an array of information for every pixel out there
"""

print("shape of X")
print (X.shape)
# #####################################
# Compute the KNN distance
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)
distances, indices = knn.kneighbors(X)
print("shape of distances")
print(distances.shape)
print ("shape of indices")
print(indices.shape)
array_dist=knn.kneighbors_graph(X).toarray()
distances = np.sort(distances, axis=0)
distances = distances[:,1]
print("shape of distances2")
print(distances.shape)

eps=distances.mean()

# Apply DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=5)
dbscan.fit(X)

print("shape of dbscan")
print(dbscan.labels_.shape)

labels = dbscan.labels_

print("shape of levels")
print(set(labels))
print(1 if -1 in set(labels) else 0)
# Count the number of clusters
num_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
print('Number of clusters:', num_clusters)

for label in labels:
    new_path=data_path/str(label)
    if new_path not in data_path.glob("*"):
        os.mkdir(new_path)


# copies images in coresponding folder
for filename, label in zip(filenames, labels):
    path=data_path/str(label)
    shutil.copy(filename,path)

