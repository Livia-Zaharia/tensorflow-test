from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.cluster import DBSCAN
from pathlib import Path
#scikit-learn

from os import listdir
from os.path import isfile, join

CURRENT_PATH_CWD = Path.cwd()
TRAINING_DATA_DIR = CURRENT_PATH_CWD/'DATA'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAIN'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAINING_DATA'

filenames_only = [f for f in listdir(TRAINING_DATA_DIR) if isfile(join(TRAINING_DATA_DIR, f))]
filenames=[]
for filename in filenames_only:
    filenames.append(TRAINING_DATA_DIR/filename)
"""
got the name of the files in the folder
"""

# Load and preprocess images
images = []
for filename in filenames:
    img = load_img(filename, target_size=(500, 250), color_mode='grayscale')
    img_array = img_to_array(img).flatten() / 255.
    images.append(img_array)

print(img_array.shape)

# Convert the list of images to a numpy array
X = np.array(images)
print (X.shape)
print(X)


# Apply DBSCAN
dbscan = DBSCAN(eps=41.5, min_samples=5)
dbscan.fit(X)


labels = dbscan.labels_

# Count the number of clusters
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print('Number of clusters:', num_clusters)

""""
# Print the cluster ID for each image
for filename, label in zip(filenames, labels):
    print(f'Image {filename} is in cluster {label}')
"""