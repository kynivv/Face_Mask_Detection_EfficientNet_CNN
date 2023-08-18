# Libraries
import pandas as pd
import numpy as np
import tensorflow as td
import os
import cv2

from zipfile import ZipFile
from tensorflow import keras
from glob import glob
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
from keras import layers


# Extracting Data From Zip
#with ZipFile('Faces.zip') as zipp:
#    zipp.extractall()


# Hyperparameters
BATCH_SIZE = 5
EPOCHS = 15
IMG_SIZE = 300
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
SPLIT = 0.25


# Data Preprocessing
X = []
Y = []

data_path = 'data'
classes = os.listdir(data_path)

for i, name in enumerate(classes):
    images = glob(f'{data_path}/{name}/*.jpg')

    for image in images:
        img = cv2.imread(image)

        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
Y = pd.get_dummies(Y)


# Data Splitting
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size= SPLIT,
                                                    shuffle= True,
                                                    random_state= 24
                                                    )


# Creating Model Checkpoint
checkpoint = ModelCheckpoint('output/model_checkpoint.h5',
                             monitor= 'val_accuracy',
                             save_best_only= True,
                             save_weights_only= True,
                             verbose= 1
                             )


# Creating Base Model
base_model = keras.applications.EfficientNetB3(include_top= True,
                                               weights= 'imagenet',
                                               input_shape= IMG_SHAPE,
                                               pooling= 'max',
                                               classes= 1000
                                               )


# Creating Model
model = keras.Sequential([
    layers.RandomRotation(30),

    base_model,

    layers.Dropout(0.1),

    layers.Dense(128, activation= 'relu'),

    layers.Dense(64, activation= 'relu'),

    layers.Dropout(0.15),

    layers.Dense(2, activation= 'softmax')
])

model.compile(optimizer= 'adam',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'],
              )


# Model Training
model.fit(X_train, Y_train,
          epochs= EPOCHS,
          batch_size= BATCH_SIZE,
          verbose= 1,
          callbacks= checkpoint,
          validation_data=(X_test, Y_test)
          )