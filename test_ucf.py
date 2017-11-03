""" Validating the model built
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
import random
import glob
from sklearn.metrics import accuracy_score
import pandas as pd

from utils import *
from c_models import *

import warnings
warnings.filterwarnings("ignore")

_labels = [i.split()[0] for i in open("txt_files/obj.names", "r")]

weight_loc = "models/model_best_weights_2.pkl"
n_classes = 101
txt_loc = 'txt_files/x.lst'

#

# Start building the model.....
model = get_model_tf_no_final(True)
x = model.output
x = Dropout(.5)(x)
predictions = Dense(n_classes, activation='softmax', name='fc8')(x)
model_final = Model(input = model.input, output = predictions)

print(model_final.summary())

#Load the weights
model_final.load_weights(weight_loc)

Actual = []
predicted = []
Prob_score = []
vid_loc = []
for num, i in enumerate(tqdm(open(txt_loc, "r"))):
    classs = i.split()[1]
    loc = i.split()[0]
    p, prob = test_vid(loc, model_final)
    Actual.append(_labels.index(classs))
    k = _labels.index(p)
    print(k, p)
    predicted.append(k)
    Prob_score.append(prob)
    vid_loc.append(loc.rsplit("/")[-1])


print("Actual", Actual)
print("Predicted", predicted)
acc_s = accuracy_score(Actual, predicted)
print("Validation Accuracy: {}".format(acc_s))

#'c3d-keras/dM06AMFLsrc.mp4'

frame = pd.DataFrame()
frame["vid_loc"] = vid_loc
frame["Actual"] = Actual
frame["Predicted"] = predicted
frame["Prob_score"] = Prob_score
frame.to_csv("txt_files/x_output.csv", index=False)
