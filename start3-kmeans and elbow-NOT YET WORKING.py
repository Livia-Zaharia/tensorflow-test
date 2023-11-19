from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.cluster import DBSCAN,KMeans
from pathlib import Path
#scikit-learn
import matplotlib.pyplot as plt
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
    img = load_img(filename, target_size=(1000, 500), color_mode='grayscale')
    img_array = img_to_array(img).flatten() / 255.
    images.append(img_array)


# Convert the list of images to a numpy array
X = np.array(images)
print (X.shape)

#it crashes from here so something might be amiss
Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k, n_init=10)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)


plt.plot(K, Sum_of_squared_distances, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Sum_of_squared_distances')
# plt.title('Elbow Method For Optimal k')
plt.show()