# TENSORFLOW-TEST
it is meant to do the following
1. take some images of lines and create clusters and thereby classes according to the shape similarity
2. be able to be fed in input other new images which to class it apartains

for point 1 we used tensorflow keras to create a RESNET model which we ulterior processed using scikit learn k nearest neighbours and dbscan to cluster. Based on the prediction from these we classified the images.

to function create a venv in which to install the requirements under python 3.10.6
to be noted tensorflow is with cuda but if your configuration doesn't allow it you might either revert to 2.14 or rebuild a custom build for gpu usage

the file test.py is there just to check if your tensorflow is gpu enabled

For Wavelet transformation procedure follow the bellow instalation process
1.Ensure that you have Python 3.6 or higher installed. You can check your Python version by running python3 --version 

2.Make sure you have the latest version of pip installed. You can upgrade pip by running 
pip install --upgrade pip 

3.Install TensorFlow if you haven't already. WaveTF requires TensorFlow 2 to be installed. You can install TensorFlow by running 
pip install tensorflow 

4.Clone the WaveTF repository from GitHub using the command 
git clone https://github.com/fversaci/WaveTF.git

5.Navigate to the cloned directory using 
cd WaveTF.

6.Finally, install WaveTF by running 
pip3 install . 

Please note that if you want to run the tests, you will also need to install pytest, numpy, and PyWavelets. You can install these using pip:

pip install pytest numpy PyWavelets
After the installation, you should be able to import and use WaveTF in your Keras scripts.
