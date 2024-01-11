# TENSORFLOW-TEST
it is meant to do the following
1. take some images of lines and create clusters and thereby classes according to the shape similarity
2. be able to be fed in input other new images which to class it apartains

for point 1 we used tensorflow keras to create a RESNET model which we ulterior processed using scikit learn k nearest neighbours and dbscan to cluster. Based on the prediction from these we classified the images.

to function create a venv in which to install the requirements under python 3.10.6
to be noted tensorflow is with cuda but if your configuration doesn't allow it you might either revert to 2.14 or rebuild a custom build for gpu usage

the file test.py is there just to check if your tensorflow is gpu enabled


for Linux/Ubuntu
python3 -m venv envlinux
source envlinux/bin/activate
pip install -r requirements.txt
