""" Manual training - Fine tuning all the layers (except the first 5)
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

from utils import *
from c_models import *

import warnings
warnings.filterwarnings("ignore")

_labels = [i.split()[0] for i in open("txt_files/obj.names", "r")]

#weight_loc = "c3d-keras/models/sports1M_weights_tf.h5"
weight_loc = "models/model_best_complete_finetune.weights"
n_classes = 101
epoch = 100
batch_size = 16

#

# Start building the model.....
model = get_model_tf_no_final(True)
x = model.output
x = Dropout(.5)(x)
predictions = Dense(n_classes, activation='softmax', name='fc8')(x)
model_final = Model(input = model.input, output = predictions)

print(model_final.summary())

#Load the weights
model_final = load_weights_manually(model_final, weight_loc)

for layer in model_final.layers[:5]:
   layer.trainable = False

#Compile the model....
lr = 0.00001
decay_rate = lr / epoch
optm = SGD(lr=lr, momentum=0.9, decay=decay_rate, nesterov=False)
model_final.compile(loss='categorical_crossentropy', optimizer=optm, metrics=['accuracy'])

print("[Model Loaded Sucessfully]")

file_loc = glob.glob("data/train/*.npy")
random.shuffle(file_loc)
file_loc_train = file_loc[0:int(0.97*len(file_loc))]
file_loc_valid = file_loc[int(0.97*len(file_loc)):]

best_acc = 0.0
for i in range(epoch)[14:]:
    random.shuffle(file_loc_train)
    flt = file_loc_train[0:16000]
    batches = int(len(flt)/batch_size)
    for j in tqdm(range(batches)):
        f = flt[j * batch_size: (j+1)*batch_size]
        x_train, y_train = npy_reader(f)
        y_train = one_hot(y_train)
        model_final.train_on_batch(x_train, y_train)
    print("Epoch {} Completed".format(i))
    Actual = []
    predicted = []
    loss = []
    for k, m in tqdm(enumerate(file_loc_valid)):
        x_valid, y_valid = npy_reader_valid(m)
        y_valid = one_hot([y_valid])
        p_loss, _ = model_final.evaluate(x_valid, y_valid, verbose = False)
        predicted.append(np.argmax(model_final.predict(x_valid)))
        Actual.append(np.argmax(y_valid))
        loss.append(p_loss)
    acc_s = accuracy_score(Actual, predicted)
    if acc_s > best_acc:
        model_final.save_weights("models/model_best_complete_finetune_2.weights")
        print("Model Saved")
        best_acc = acc_s
    print("Validation Accuracy: {}, Validation Loss:{}".format(acc_s, sum(loss)/k))
