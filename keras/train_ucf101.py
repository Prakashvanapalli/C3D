""" Building your own model on UCF-101

Author @Prakash

References used:
https://gist.github.com/jfsantos/e2ef822c744357a4ed16ec0c885100a3

"""
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras.utils.io_utils import HDF5Matrix
import h5py

from utils import *
from c_models import *

import warnings
warnings.filterwarnings("ignore")

weight_loc = "c3d-keras/models/sports1M_weights_tf.h5"
n_classes = 1
train_loc = 'data/train.h5'
valid_loc = 'data/valid.h5'

# # Load the train Dataset
# X_train = HDF5Matrix(train_loc, 'images', start=0, normalizer=preprocess_train)
# y_train = HDF5Matrix(train_loc, 'labels')
#
# # Likewise for the test set
# X_test = HDF5Matrix(valid_loc, 'images',start=0, normalizer=preprocess_valid)
# y_test = HDF5Matrix(valid_loc, 'labels')

X_train = HDF5Matrix("data/train_1.h5", 'images')
y_train = HDF5Matrix(train_loc, 'labels')


# Start building the model.....
model = get_model_tf_no_final(True)
x = model.output
x = Dropout(.5)(x)
predictions = Dense(n_classes, activation='sigmoid', name='fc8')(x)
model_final = Model(input = model.input, output = predictions)

print(model_final.summary())

#Load the weights
model_final = load_weights_manually(model_final, weight_loc)

#Compile the model....
model_final.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("[Model Loaded Sucessfully]")


## Training the model
#model_final.fit(X_train, y_train, batch_size=16, shuffle='batch', validation_data = [X_test, y_test], epochs=10)
model_final.fit(X_train, y_train, batch_size=16, shuffle='batch', epochs=10)
