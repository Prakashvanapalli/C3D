""" Given an Untrimmed Video. Gives the output of when the mopping action is taking place
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
import shutil

from utils import *
from c_models import *

import warnings
warnings.filterwarnings("ignore")

_labels = [i.split()[0] for i in open("txt_files/obj.names", "r")]

def load_built_model(weight_loc, n_classes):
    model = get_model_tf_no_final(True)
    x = model.output
    x = Dropout(.5)(x)
    predictions = Dense(n_classes, activation='softmax', name='fc8')(x)
    model_final = Model(input = model.input, output = predictions)
    # print(model_final.summary())
    return model_final.load_weights(weight_loc)



def video_chunks_ffpmeg(vid_loc, save_loc, seconds=10):
    if os.path.exists(save_loc):
        shutil.rmtree(save_loc)

    if not os.path.exists(save_loc):
        os.makedirs(save_loc)
    video_save_loc = save_loc+vid_loc.rsplit("/")[-1].rsplit(".")[0]+"_"
    k = "ffmpeg -i {} -c copy -map 0 -segment_time {} -f segment -reset_timestamps 1 {}%03d.mp4".format(vid_loc, seconds, video_save_loc)
    os.system(k)


if __name__ == '__main__':
    weight_loc = "models/model_best_weights_2.pkl"
    n_classes = 101

    folder_loc = "../.."

    k = []
    k_prob = []
    k_cat = []
    test_videos = ["xx.mp4"]




    for tv in test_videos:
        print("[Dividing the video into chuncks of 10 secs]")
        video_chunks_ffpmeg(folder_loc+tv, "temp/", 10)

        vid_avail = glob.glob("temp/*.mp4")
        print("the total_number of video chunks in the folder are: {}".format(len(vid_avail)))

        print("Model Loading")
        #model_final = load_built_model(weight_loc, n_classes)

        model = get_model_tf_no_final(True)
        x = model.output
        x = Dropout(.5)(x)
        predictions = Dense(n_classes, activation='softmax', name='fc8')(x)
        model_final = Model(input = model.input, output = predictions)
        # print(model_final.summary())
        model_final.load_weights(weight_loc)

        for i in vid_avail:
            p, prob = test_vid(i, model_final)
            k.append(i)
            k_prob.append(prob)
            k_cat.append(p)
            print(i,"_ ",  p, "_", prob)

        print("[Completed {}]".format(tv))

    frames = pd.DataFrame()
    frames["Video_ID"] = k
    frames["Prob"] = k_prob
    frames["Category"] = k_cat
    frames.to_csv("txt_files/test.csv", index=False)

    ## threshold-0.98 will remove all false positives
