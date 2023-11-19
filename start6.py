from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import os
import glob
import shutil
import plotly as plt
import tensorflow as tf

#scikit-learn, opnecv-python

from os import listdir
from os.path import isfile, join

CURRENT_PATH_CWD = Path.cwd()
TRAINING_DATA_DIR = CURRENT_PATH_CWD/'DATA'
data_path=TRAINING_DATA_DIR
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAIN'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAINING_DATA'

filenames_only = [f for f in listdir(TRAINING_DATA_DIR) if isfile(join(TRAINING_DATA_DIR, f))]
filenames=[]
for filename in filenames_only:
    filenames.append(TRAINING_DATA_DIR/filename)
"""
got the name of the files in the folder

"""

import cv2
print(filenames[0])
# Load an image
image = cv2.imread(str(filenames[0]),0)
print("cu grayscale")
print(image)
print(image.shape)

im = cv2.imread(str(filenames[0]))
print("cu rgb")
print(im)
print(im.shape)

####################

im = np.expand_dims(im, 0)
print("cu rgb dupa expansion")
print(im)
print(im.shape)

MyModel = tf.keras.models.Sequential()
MyModel.add(tf.keras.applications.ResNet101(
    include_top = False, weights='imagenet', pooling='avg')
    )

# freezing weights for 1st layer
MyModel.layers[0].trainable = False


img = tf.keras.applications.resnet50.preprocess_input(im)
extractedFeatures = MyModel.predict(img)
extractedFeatures = np.array(extractedFeatures)

print("features")
print(extractedFeatures)
print(extractedFeatures.shape)
print("features cu flatten")
print(extractedFeatures.flatten())
print(extractedFeatures.flatten().shape)


"""

# Apply Canny edge detection
edges = cv2.Canny(image, 100, 200)

# Display the result
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Flatten the edge map into a 1D array
edges_flat = edges.flatten()/255

print('openCV')
print(edges_flat.shape)
print(edges_flat)
af=[]
for i,j in enumerate (edges_flat):
    if (i%1000)<1000 and i%1000 !=0:
        af.append(j)
    else:
        print(af)
        af=[]
    


img = load_img(filenames[0], target_size=(1000, 500), color_mode='grayscale')
img_array = img_to_array(img).flatten() / 255.

print('keras')
print(img_array.shape)
print(img_array)

"""