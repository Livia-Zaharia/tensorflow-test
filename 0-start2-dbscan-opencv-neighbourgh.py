import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import os
import glob
import shutil
import cv2

CURRENT_PATH_CWD = Path.cwd()
TRAINING_DATA_DIR = CURRENT_PATH_CWD/'DATA'
data_path=TRAINING_DATA_DIR
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAIN'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAINING_DATA'

filenames_only = [f for f in os.listdir(TRAINING_DATA_DIR) if os.path.isfile(os.path.join(TRAINING_DATA_DIR, f))]
filenames=[]
for filename in filenames_only:
    filenames.append(TRAINING_DATA_DIR/filename)
"""
got the name of the files in the folder
"""

# Load and preprocess images
images = []
for filename in filenames:
    # Load an image
    img = cv2.imread(str(filename), 0)

    # let's downscale the image using new  width and height
    down_width = 250
    down_height = 125
    down_points = (down_width, down_height)
    resized_down = cv2.resize(img, down_points, interpolation= cv2.INTER_LINEAR)

    # Apply Canny edge detection
    edges = cv2.Canny(resized_down, 100, 200)

    # # Display the result
    # cv2.imshow('Edges', edges)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Flatten the edge map into a 1D array
    edges_flat = edges.flatten()/255
    
    images.append(edges_flat)

# print(edges_flat.shape)

# Convert the list of images to a numpy array
X = np.array(images)
# print (X.shape)
# print(X)

# Compute the KNN distance
knn = NearestNeighbors(n_neighbors=2)
knn.fit(X)
distances, indices = knn.kneighbors(X)
array_dist=knn.kneighbors_graph(X).toarray()
distances = np.sort(distances, axis=0)
distances = distances[:,1]

eps=distances.mean()-2

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