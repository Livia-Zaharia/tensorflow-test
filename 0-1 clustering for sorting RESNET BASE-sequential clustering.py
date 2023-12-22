
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


sk_struct = {
    'photo_name' : [],
    'flattenPhoto' : []
    }



MyModel=tf.keras.applications.ResNet101(
    include_top = False, weights='imagenet', pooling='avg')


MyModel.trainable=False

MyModel.compile(optimizer=tf.keras.optimizers.Adam())

MyModel.trainable=True

MyModel.layers[-1].trainable = False
MyModel.compile(optimizer=tf.keras.optimizers.Adam())


"""
Here ends the model as defined by resnet
"""





def LoadDataAndDoEssentials(path, h, w):
    """
    Part2/2 of the basic setup used to process the images. It is called by part 1/2
    """
    
    img = cv2.imread(str(path))
    img = cv2.resize(img, (h, w))
    
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)

    img = tf.keras.applications.resnet50.preprocess_input(img)
   
    extractedFeatures = MyModel.predict(img)
        
    extractedFeatures = np.array(extractedFeatures)

    #THIS IS THE IMPORTANT ROW IN THIS PART
    sk_struct['flattenPhoto'].append(extractedFeatures.flatten())
    

def ReadAndStoreMyImages(path):
    """
    Part1/2 of the basic setup used to process the images. 
    It is called from main at the moment and sets up the looping
    """

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
        
        
def filter_the_array(x_val, filter_val, filter_list):
    return np.fromiter((x for (y,x) in enumerate(x_val) if filter_list[y]==filter_val),
                      dtype = 'object') 



def compute_dbscan_labels(object_x):
    new_x = np.vstack(object_x[:]).astype(np.float64)
    
    # Compute the KNN distance
    knn.fit(new_x)
    new_distances, new_indices = knn.kneighbors(new_x)

    new_distances = np.sort(new_distances, axis=0)
    new_distances = new_distances[:,1]

    new_eps=new_distances.mean()

    # Apply DBSCAN
    new_dbscan = DBSCAN(eps=new_eps, min_samples=5)
    new_dbscan.fit(new_x)

    new_labels = new_dbscan.labels_
    new_unique_labels=set(new_labels)
    print(new_unique_labels)
 




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


print("shape of X")
print (X.shape)

# Compute the KNN distance
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)
distances, indices = knn.kneighbors(X)
print("shape of distances")
print(distances.shape)
print ("shape of indices")
print(indices.shape)

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
unique_labels=set(labels)


# Count the number of clusters
num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
print('Number of clusters:', num_clusters)



for label in unique_labels:
    new_x=filter_the_array(X,label,labels)
    compute_dbscan_labels(new_x)

"""

print("new-x dtype",float_arr.dtype)
print("new-x shape", float_arr.shape)
    
print("new-x element dytpe", type(float_arr[0]))
print(len(float_arr[0]))
print(float_arr[0])

print("x dtype",X.dtype)
print("x shape", X.shape)
print("x element dytpe", type(X[0]))
print(len(X[0]))
print(X[0])

for label in unique_labels:
    new_path=data_path/str(label)
    if new_path not in data_path.glob("*"):
        os.mkdir(new_path)


# copies images in coresponding folder
for filename, label in zip(filenames, labels):
    path=data_path/str(label)
    shutil.copy(filename,path)

"""