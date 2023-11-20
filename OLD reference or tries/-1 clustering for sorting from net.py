# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



###########################################################################
# Solution NoteBook for the Problem Keep babies save                      #
###########################################################################
##################### Making Essential Imports ############################
import sklearn
import os
import sys
import matplotlib.pyplot as plt
import cv2
import pytesseract
import numpy as np
import pandas as pd
import tensorflow as tf
conf = r'-- oem 2'
#####################################
# Defining a skeleton for our       #
# DataFrame                         #
#####################################

DataFrame = {
    'photo_name' : [],
    'flattenPhoto' : [],
    'text' : [],

    }
#######################################################################################
#      The Approach is to apply transfer learning hence using Resnet50 as my          #
#      pretrained model                                                               #
#######################################################################################

MyModel = tf.keras.models.Sequential()
MyModel.add(tf.keras.applications.ResNet101(
    include_top = False, weights='imagenet',    pooling='avg',
))

# freezing weights for 1st layer
MyModel.layers[0].trainable = False
### Now defining dataloading Function
def LoadDataAndDoEssentials(path, h, w):
    img = cv2.imread(path)
    DataFrame['text'].append(pytesseract.image_to_string(img, config = conf))
    img = cv2.resize(img, (h, w))
    ## Expanding image dims so this represents 1 sample
    img = img = np.expand_dims(img, 0)
    
    img = tf.keras.applications.resnet50.preprocess_input(img)
    extractedFeatures = MyModel.predict(img)
    extractedFeatures = np.array(extractedFeatures)
    DataFrame['flattenPhoto'].append(extractedFeatures.flatten())
    
### with this all done lets write the iterrrative loop
def ReadAndStoreMyImages(path):
    list_ = os.listdir(path)

    for mem in list_:
        DataFrame['photo_name'].append(mem)
        imagePath = path + '/' + mem
        LoadDataAndDoEssentials(imagePath, 224, 224)
### lets give the address of our Parent directory and start
path = '/kaggle/input/keep-babies-safe/dataset/images'
ReadAndStoreMyImages(path)
######################################################
#        lets now do clustering                      #
######################################################

Training_Feature_vector = np.array(DataFrame['flattenPhoto'], dtype = 'float64')
from sklearn.cluster import AgglomerativeClustering
kmeans = AgglomerativeClustering(n_clusters = 2)
kmeans.fit(Training_Feature_vector)
AgglomerativeClustering()
predictions = kmeans.labels_
NamePred = []
for mem in predictions:
    if mem == 0:
        NamePred.append('toys')
    else:
        NamePred.append('consumer_products')
textAns = np.array(DataFrame['text'])
realText = []
import re

for mem in textAns:
    newMem = re.sub("\s\s+", " ", mem)
    if len(newMem) == 0 or newMem == " ":
        realText.append('Unnamed')
        continue
    else:
        realText.append(str(newMem))
names = DataFrame['photo_name']
df = {
    'Image' : names,
    'Class_of_image' : NamePred,
    'Brand_name' : realText
}
df = pd.DataFrame(df)
df.to_csv('predictions2.csv', index = False)
df.head(50)
