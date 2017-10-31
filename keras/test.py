from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D, ZeroPadding3D
from keras.optimizers import SGD
from c_models import *
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils import *

import warnings
warnings.filterwarnings("ignore")


model = get_model_tf(summary=True)
model.load_weights("c3d-keras/models/sports1M_weights_tf.h5")
print("[Model Loaded Sucessfully]")

cap = cv2.VideoCapture('c3d-keras/dM06AMFLsrc.mp4')
vid = []
while True:
    ret, img = cap.read()
    if not ret:
        break
    #print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    vid.append(cv2.resize(img, (171, 128)))

vid = np.array(vid, dtype=np.float32)
print(vid.shape)
print("[Video Captured Sucessfully]")

print("[Loading the mean weights]")
mean_cube = np.load('c3d-keras/models/train01_16_128_171_mean.npy')
mean_cube = np.transpose(mean_cube, (1, 2, 3, 0))
print(mean_cube.shape)


# start_frame = 2000
# X = vid[start_frame:(start_frame + 16), :, :, :]
output = []
for f in tqdm(gen_frame(vid.shape[0], 16, 8)):
    X = np.concatenate([vid[i][np.newaxis, :, :, :] for i in f])
    X -= mean_cube
    X = X[:, 8:120, 30:142, :]
    m = model.predict(X[np.newaxis, :, :, :, :])
    output.append(m)

output = np.concatenate(output)
output = np.argmax(output.mean(axis=0))
print(output)


# X -= mean_cube
# X = X[:, 8:120, 30:142, :] # (l
# print("Final Video Array:", X.shape)
# print(np.argmax(model.predict(X[np.newaxis, :, :, :, :])))
