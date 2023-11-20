#import matplotlib.pyplot as plt
import os
import tensorflow as tf
from pathlib import Path
#pillow

#import matplotlib
#matplotlib.style.use('ggplot')

'''
just to be clear- to work the imagedata geneerator needs a path that has a folder_0, containing folders_1 and_2 with the images. 
the path has to reference folder_0
'''

IMAGE_SHAPE = (1000, 500)
CURRENT_PATH_CWD = Path.cwd()
TRAINING_DATA_DIR = CURRENT_PATH_CWD/'DATA'
TRAINING_DATA_DIR = TRAINING_DATA_DIR/'TRAIN'


VALID_DATA_DIR =CURRENT_PATH_CWD/'DATA'
VALID_DATA_DIR =VALID_DATA_DIR/'VALID'

print(TRAINING_DATA_DIR)
print(VALID_DATA_DIR)

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

print (datagen)
train_generator = datagen.flow_from_directory(
    TRAINING_DATA_DIR,
    shuffle=True,
    target_size=IMAGE_SHAPE
)


valid_generator = datagen.flow_from_directory(
    VALID_DATA_DIR,
    shuffle=False,
    target_size=IMAGE_SHAPE
)
"""

batch_size = 32
img_height = 500
img_width = 1000

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

"""

def build_model(num_classes):
    model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu', 
                           input_shape=(1000, 500, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model
model = build_model(num_classes=1)


model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
print(model.summary())

EPOCHS = 20
BATCH_SIZE = 32
history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps= valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )