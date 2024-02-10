
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
import time

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from wavelet import WaveletTransformLayer as wtl

CURRENT_PATH_CWD = Path.cwd()
data_path=CURRENT_PATH_CWD/'DATA'
TRAINING_DATA_DIR = data_path/'TRAIN'/'TRAINING_DATA'

def main():
    
    create_global_structures()
    
    read_and_store_img(TRAINING_DATA_DIR)

    ################################################# CLUSTERING

    X = np.array(sk_struct['flattenPhoto'], dtype = 'float64')

    labels = clustering(X)
    unique_labels=set(labels)

    ######### Count the number of clusters
    num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    
    #########################RECURSIVE classification
    #basically- if we already split the items into classes- but those are not yet distinct- 
    # we shall run the prgram recusively until only one class emerges
    
    data_value, label_value,filename_value=splitting(X,filenames,unique_labels,labels)

    #######################################################################
    #from here we have the images divided in their folder- what we need is 
    # reencoding for the big picture and then rerun of knn and dbscan
    
    reencoded=reencoding(TRAINING_DATA_DIR, filename_value,label_value)
    
    for label in set(label_value):
        new_path=data_path/str(label)
        if new_path not in data_path.glob("*"):
            os.mkdir(new_path)


    # copies images in coresponding folder
    for filename, label in zip(filename_value, label_value):
        path=data_path/str(label)
        shutil.copy(str(filename[0]),path)
        

    




def create_global_structures():
    """
    Defines some global structures used in the whole script. They are centralized here
    """
    
    #strucutre in which to hold the image and its name
    global sk_struct
    sk_struct = {
        'photo_name' : [],
        'flattenPhoto' : []
        }
    
    #the resnet and wavelet models
    create_models()
    
    #the filenames list where the names are stored after the recursive
    global filenames
    filenames=[]

def create_models():
    """
    Define the models. There are only two of them
    The Resnet model- comes from tf.keras.applications- and because it is not defined from 
    tf.keras.model.sequential it is taken as a functional one
    
    the wavelet model- created because the wtl class only creates a layer- not a model
    the model is also a functional API one and distills the the resnet
    """
    ############### RESNET MODEL DEFINITION
    
    global MyModel
    MyModel=tf.keras.applications.ResNet101(
        include_top = False, weights='imagenet',pooling='avg')

    MyModel.trainable=False

    MyModel.compile(optimizer=tf.keras.optimizers.Adam())


    ################# ADAPTATIVE WAVELETS
    
    global model
    # Apply the custom wavelet transform layer to the output tensor

    wavelet_transform = wtl()(MyModel.output)

    # Create a new model that includes the wavelet transform layer
    model = tf.keras.models.Model(inputs=MyModel.input, outputs=wavelet_transform)
    
    ###################KNN
    
    global knn
    knn = NearestNeighbors(n_neighbors=2)

def clustering (X):
    ############################# Compute the KNN distance
    
    knn.fit(X)
    distances, indices = knn.kneighbors(X)
    #the model of knn returns all the distances of the closest 2points and the indices (who they were)

    distances = np.sort(distances, axis=0)
    distances = distances[:,1]

    eps=distances.mean()

    ############# Apply DBSCAN
    #by knowing that avreage distance that the neighbours were found on we use DSCAN to cluster the whole lot and make labels
    
    dbscan = DBSCAN(eps=eps, min_samples=5)
    dbscan.fit(X)

    labels = dbscan.labels_
    return labels



def extract_filenames(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def read_and_store_img(path):
    """
    Part1/2 of the basic setup used to process the images.
    It reads all the names of the files in the directory
    and populates the structure of dict in a loop
    calls upon the properties and data
    """

    filenames_only = extract_filenames(path)
    i=0
    ##############################################
    #here starts the addition
    #here introduced so I can use the presplit images- just to keep in mind
    new_path=data_path/'SPLIT'
    
    # creates a folder and a new vector to store names if they exist
    if new_path not in data_path.glob("*"):
        os.mkdir(new_path)
        
    new_filenames_only = extract_filenames(new_path)
    ####################################################
    
    
    ##############################################
    
    #if the images were not split it comes here    
    if len(new_filenames_only)==0:
    
        for filename in filenames_only:
            split_image(filename,path,4,2,new_path)
            #it does not produce an output because it creates a folder with images- acts on the outside
    ###############################################
    
        
    filenames_only=[]
    filenames_only=new_filenames_only    
        
    ########################################################################
    #and thus from here it gets in the normal workflow
        
    for filename in filenames_only:
        #THIS IS THE IMPORTANT ROW IN THIS PART
        sk_struct['photo_name'].append(filename)
        print(i)
        i+=1
        imagePath=new_path/filename
        filenames.append(imagePath)
        load_data_and_basic_ops(imagePath, 224, 224)
       
        '''
        if i>500: 
            break
        else: i=i+1
        print(filename)
       '''
        
def load_data_and_basic_ops(path, h, w):
    """
    Part2/2 of the basic setup used to process the images. It is called by part 1/2
    Does the first processing of the images- basically transposes images into a vector 
    """
    
    img = cv2.imread(str(path))
    img = cv2.resize(img, (h, w))
            
    ## Expanding image dims so this represents 1 sample
    img = np.expand_dims(img, 0)
       
    img = tf.keras.applications.resnet50.preprocess_input(img)  
    extractedFeatures = MyModel.predict(img)
    #here we get the output from the Resnet model

     
    filter1,filter2=model.predict(img)
    #here we get the output from the wavelet model
        
    extractedFeatures = np.array(extractedFeatures)
    #conversion to np array- from the Resnet we wil get something along the
    # line shape (1,2048)- basically the values in each pixel for each image

    
    extractedFeatures=np.add(extractedFeatures,filter1)
    extractedFeatures=np.add(extractedFeatures,filter2)
    #adding the results from the wavelet in the array. the wavelet model returns 
    # two indices which are converte in the wavelet class to be of similar shape

    #THIS IS THE IMPORTANT ROW IN THIS PART
    sk_struct['flattenPhoto'].append(extractedFeatures.flatten())

def split_image(filename,path,w_size,h_size,new_path):
    """
    It only activiates if the data is not split already so it won't be 
    found in other places but in read_and_store
    """
    img=cv2.imread(str(path/filename))
    
    height, width, channels = img.shape

    for ih in range(h_size ):
        for iw in range(w_size ):
   
            x = width/w_size * iw 
            y = height/h_size * ih
            h = (height / h_size)
            w = (width / w_size )
            
            img2 = img[int(y):int(y+h), int(x):int(x+w)]
            name=str(filename[:-4])+"-"+str(ih)+str(iw)+".png"
            cv2.imwrite(os.path.join(new_path,name),img2)

     
     
     
def splitting(c_X,c_path,c_unique_labels,c_labels):
    count_start=0
    for label in c_unique_labels:
        new_x=filter_the_array(c_X,label,c_labels)
        new_paths=filter_the_array(c_path,label,c_labels)
        some_x, some_labels, some_paths=compute_dbscan_labels(new_x,new_paths,label)
       
       #this whole part from here genrates the continuity of numbering
        if -1 in some_labels:
            some_labels=some_labels+1
       
        some_labels=some_labels+count_start
        
        if count_start==0:
            prev_x=some_x
            prev_label=some_labels
            prev_path=some_paths
        else:
            prev_x=np.concatenate((prev_x,some_x),axis=0)
            prev_label=np.concatenate((prev_label,some_labels),axis=0)
            prev_path=np.concatenate((prev_path,some_paths),axis=0)
        
        count_start= max(some_labels)+1
    
    return (prev_x,prev_label, prev_path)
                 
def filter_the_array(x_val, filter_val, filter_list):
    return np.fromiter((x for (y,x) in enumerate(x_val) if filter_list[y]==filter_val),
                      dtype = 'object') 

def compute_dbscan_labels(object_x, path_x, origin_label):
    new_x = np.vstack(object_x[:]).astype(np.float64)
    new_path = np.vstack(path_x[:]).astype(str)
    
    '''
    if len(set(map(tuple,new_x)))!=1:
        new_labels = clustering(new_x)
    else:
        new_labels=np.full(new_x.size,origin_label)
        
    '''
    new_labels = clustering(new_x)
    print(new_labels)
    print(type(new_labels))
    
        
    new_unique_labels=set(new_labels)
    
    if len(new_unique_labels) != 1:
        return splitting(new_x,new_path,new_unique_labels,new_labels)
    else:
        return (new_x, new_labels,new_path)



def reencoding(path, filename_value, label_value):
    
    re={}
    names_single=extract_filenames(path)
    for title in names_single:
        re.setdefault(title,[])
        
    for name, label in zip (filename_value, label_value):
        origin=str(name).split('/')
        o=origin[-1]
        o1=o[:-9]+".png"
        o=o[-8:-6]
        
        re[o1].append((int(o),label))
     
    sorted_re={}  
    for i in names_single:
        sorted_re.setdefault(i,sorted(re[i], key=lambda x: x[0]))
     
    re.clear
    re = {k: [x[1] for x in v] for k, v in sorted_re.items()}
    
    #print (re)
    #print("++++++++++")
    return re
        
        
        


if __name__ == "__main__":
    main()