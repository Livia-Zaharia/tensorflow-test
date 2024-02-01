# TENSORFLOW-TEST

#
## Idea

The main objectives are:
1. take some images of graphs and create clusters and thereby classes according to the shape similarity

1.1 implies taking the image and divide it in smaller sections
1.2 these sections are the ones clustered because having a simpler form it is easier to run image recognition on them
1.3 the original image is re-encoded according to these classes
1.4 final similarity per image is determined

2. be able to be fed in input other new images which to sort and classify

#
## The Logic

For objective 1

1.
Tensorflow.keras was used to create a ResNet model for the first processing.
After the wavelets model basically does a mathematical aproximation of the shape into a wave.

2.
Scikit-learn was used with k-Nearest Neighbours and DSCAN to cluster the results provided from the model.predict ()
The number of classes estimated from the Dbscan is checked.

3.
The sequential recursive clustering basically means each determined class is going through the clustering procedure until it will return only one class. 
So step 2 is being applied recursively until there is just one class returned

4.
Further steps are yet to be implemented but it will have to deal with the final sorting of the complete images and re encoding

For objective 2
Not yet developed

#
## Requirements

It was first developed on Windows and ran with Tensorflow on CPU. It can still do that but will take long hours.
At the moment it will be developed under WSL/Linux so that it can run on Tensorflow on GPU with a custom build of Tensorflow

In either case, for both operating system there should be a virtual enviroment created with python 3.10.6

If by chance anyone wondered why there has to be two separate enviorments for the different OS found a good answer [here](https://stackoverflow.com/questions/42733542/how-to-use-the-same-python-virtualenv-on-both-windows-and-linux)


#
## Windows

### Setting up the enviorment
`python -m venv .venv`

### Activating the enviorment
`.venv\Scripts\activate`

### Deactivating the enviorment
`deactivate`

### Installing the requirements
`pip install -r requirements. txt`

 


#
## Linux

### Setting up the enviorment
`python3 -m venv envlinux`

### Activating the enviorment
`source envlinux/bin/activate`

### Deactivating the enviorment
`deactivate`

### Installing the requirements
`pip3 install -r /path/to/requirements.txt`


The file test.py is there just to check if your tensorflow is gpu enabled
If not you can either run on CPU or build your own Tensorflow from source

For that purpose, follow the instructions in 
REBUILDING FROM SOURCE OF TENSORFLOW