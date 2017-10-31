""" Utils file
"""
import numpy as np
import h5py
import cv2
import os
from ir.feature_indexer import FeatureIndexer
from tqdm import tqdm


_labels = [i.split()[0] for i in open("txt_files/obj.names", "r")]
mean_file = np.load("data/mean_data.npy")[np.newaxis, :, :, :, :]

def gen_frame(nframes, frames=16, stride=8):
    """ Generate list of list acording to frames and stride
     Ex: nframes=16, frames=4, stride=2
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10, 11, 12, 13, 14, 15]

     [[0, 1, 2, 3], [2, 3, 4, 5], [4, 5, 6, 7], [6, 7, 8, 9] .....]

    """
    last_num = nframes - (nframes%stride)
    total_batches = int(last_num /stride)
    for i in range(total_batches-1):
        m = i * stride
        n = (frames) + (i * stride)
        x = list(range(m, n))
        yield x

def read_vid(vid_loc):
    cap = cv2.VideoCapture(loc)
    vid = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        vid.append(cv2.resize(img, (171, 128)))
    vid = np.array(vid, dtype=np.float32)
    return vid

def load_weights_manually(model, weight_loc):
    f = h5py.File(weight_loc, "r")
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_names = [layer.name for layer in model.layers]
    for i in f.keys():
        if i == "fc8": continue
        weight_names = f[i].attrs["weight_names"]
        weights = [f[i][j] for j in weight_names]
        index = layer_names.index(i)
        model.layers[index].set_weights(weights)
    return model

def create_h5py_train_dataset(txt_loc, formats=".mp4", db_name="data/train.h5"):
    num_lines = sum(1 for line in open(txt_loc))
    fi = FeatureIndexer(db_name, num_lines, 64)
    for i in tqdm(open(txt_loc, "r")):
        classs = i.split()[1]
        loc = i.split()[0]
        cap = cv2.VideoCapture(loc)
        vid = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            #print(img.shape)
            vid.append(cv2.resize(img, (171, 128)))
        label = _labels.index(classs)
        for fra in gen_frame(len(vid), 16, 8):
            X = np.concatenate([vid[i][np.newaxis, :, :, :] for i in fra])
            X = X[:, 8:120, 30:142, :]/255.0
            fi.add(X[np.newaxis, :, :, :, :], label)
    fi.finish()


def preprocess_train(array):
    """ Given a batch of numpy arrays, it outputs a batch of numpy of arrays with all preprocessing

    size : (w, h)
    """
    num1 = np.random.randint(0, 128 - 112)
    num2 = np.random.randint(0, 171 - 112)
    crop = array[ :, num1:num1+112, num2:num2+112, :]
    crop = crop/255.0
    return  crop

def preprocess_valid(array):
    """ Given a batch of numpy arrays, it outputs a batch of numpy of arrays with all preprocessing

    size : (w, h)
    """
    crop = array[ :, 8:120, 30:142, :]
    crop = crop/255.0
    return  crop


def create_npy_train_dataset(txt_loc, formats=".mp4", db_name="data/train/"):
    num_lines = sum(1 for line in open(txt_loc))
    for k, i in tqdm(enumerate(open(txt_loc, "r"))):
        classs = i.split()[1]
        loc = i.split()[0]
        cap = cv2.VideoCapture(loc)
        vid = []
        while True:
            ret, img = cap.read()
            if not ret:
                break
            #print(img.shape)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            vid.append(cv2.resize(img, (171, 128)))
        label = _labels.index(classs)
        for f, fra in enumerate(gen_frame(len(vid), 16, 8)):
            X = np.concatenate([vid[m][np.newaxis, :, :, :] for m in fra])
            X = X[np.newaxis, :, :, :, :]
            np.save(db_name+str(k)+"_"+str(f)+"_"+classs+".npy", X, allow_pickle=True)

def one_hot(batch_labels):
    nb_classes = len(_labels)
    targets = np.array([_labels.index(i) for i in batch_labels])
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets

def random_crop(array):
    num1 = np.random.randint(0, 128 - 112)
    num2 = np.random.randint(0, 171 - 112)
    crop = [ik[ :, :,  num1:num1+112, num2:num2+112, :] for ik in array]
    return  np.concatenate(crop)

def npy_reader(f):
    labels = [i.rsplit("/")[-1].rsplit(".")[0].rsplit("_")[-1] for i in f]
    f = [np.load(i) - mean_file for i in f]
    f = random_crop(f)
    return (f, labels)

def npy_reader_valid(f):
    labels = f.rsplit("/")[-1].rsplit(".")[0].rsplit("_")[-1]
    f = np.load(f) - mean_file
    f = f[ :, :,  8:120, 30:142, :]
    return (f, labels)

if __name__ == '__main__':
    if not os.path.exists("data/train/"):
        os.makedirs("data/train/")
    # create_h5py_train_dataset("txt_files/train.txt", ".avi", "data/train_1.h5")
    # create_h5py_train_dataset("txt_files/valid.txt", ".avi", "data/valid_1.h5")
    create_npy_train_dataset("txt_files/train.lst", ".avi", db_name="data/train/")

    if not os.path.exists("data/valid/"):
        os.makedirs("data/valid/")
    create_npy_train_dataset("txt_files/valid.lst", ".avi", db_name="data/valid/")
