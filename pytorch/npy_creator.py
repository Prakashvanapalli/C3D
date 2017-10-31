""" Preparing dataset for training DL models, creating train, valid and test npy files
"""

import numpy as np
import imageio
imageio.plugins.ffmpeg.download()
import PIL
from PIL import Image
import os
import random
import linecache
from ir.feature_indexer import FeatureIndexer
from tqdm import tqdm


_labels = [i.split()[0] for i in open("txt_files/obj.names", "r")]

def one_hot(batch_labels):
    nb_classes = len(_labels)
    targets = np.array([_labels.index(i) for i in batch_labels])
    #one_hot_targets = np.eye(nb_classes)[targets]
    return targets

def preprocess_one_batch(array, resize=True, size=(171, 128), random_crop=True, crop_size= (112, 112)):
    """ Given a batch of numpy arrays, it outputs a batch of numpy of arrays with all preprocessing

    size : (w, h)
    """
    if resize:
        pil_images = [Image.fromarray(i.astype('uint8'), 'RGB') for i in array]
        pil_images = [i.resize(size, Image.BILINEAR) for i in pil_images]
        images = [np.array(i) for i in pil_images]
    else:
        images = [i for i in array]

    if random_crop:
        num1 = np.random.randint(0, size[1] - crop_size[1])
        num2 = np.random.randint(0, size[0] - crop_size[0])
        crop = [ik[ num1:num1+crop_size[1], num2:num2+crop_size[0], :] for ik in images]

    else:
        crop = images

    return  np.concatenate([i[np.newaxis, :, :, :] for i in crop])

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

def gen_vid_frames(vid_loc, frames=16, stride=8, formats=".mp4", verbose=True):
    """ Given a video_loc generate the frames
    """
    vid = imageio.get_reader(vid_loc, formats)
    nframes = vid.get_meta_data()["nframes"]
    if verbose:
        print("vid_loc: ", vid_loc, "total_frames: ", nframes, " fps: ", vid.get_meta_data()["fps"] )

    frames = gen_frame(nframes, frames, stride)
    for f in frames:
        img_list = [vid.get_data(k) for k in f]
        img_list = [k[np.newaxis, :, :, :] for k in img_list]
        yield np.concatenate(img_list)

def gen_dataset(txt_loc, frames=16, stride=8, formats=".mp4", verbose=False):
    """ split each video as a set of numpy files and save it in respective folders
    """
    for i in open(txt_loc, "r"):
        vid_loc = i.split()[0]
        vid_frames = gen_vid_frames(vid_loc, frames, stride, formats, verbose)
        for fra in vid_frames:
            yield (fra, vid_loc)

def train_one_vid_frames(vid_loc, start_frame, end_frame, formats=".mp4", resize = True, size = (171, 128), random_crop = True, crop_size= (112, 112)):
    """ It returns the desired number of frames from a video, given video location
    """
    vid = imageio.get_reader(vid_loc, formats)
    img_list = [vid.get_data(k) for k in range(start_frame, end_frame+1)]
    #img_list = [k[np.newaxis, :, :, :] for k in img_list]
    img_list = preprocess_one_batch(img_list, resize, size, random_crop, crop_size)
    return img_list

def train_batch_vid_frames(batch_list, formats=".mp4", resize = True, size = (171, 128), random_crop = True, crop_size= (112, 112)):
    for i in batch_list:
        vid_loc = i[0]
        start_frame = int(i[1])
        end_frame = int(i[2])
        yield train_one_vid_frames(vid_loc, start_frame, end_frame, formats, resize, size, random_crop, crop_size)[np.newaxis, :, :, :]

def chunks(l, n):
    """Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]

def train_gen(txt_loc , batch_size = 30, formats=".mp4", resize = True, size = (171, 128), random_crop = True, crop_size= (112, 112)):
    """Generates a numpy array of random images to train the DNN
    """
    num_lines = sum(1 for line in open(txt_loc))
    print("total ids", num_lines)
    list_of_num_lines = list(range(0, num_lines))
    random.shuffle(list_of_num_lines)

    for chunk in chunks(list_of_num_lines, batch_size):
        try:
            if len(chunk) != batch_size: continue
            line = [linecache.getline(txt_loc,i) for i in chunk]
            line = [i.split() for i in line]
            label = [i[0].rsplit("/")[-2] for i in line]
            one_hot_labels = one_hot(label)
            vid_batch = list(train_batch_vid_frames(line, formats, resize, size, random_crop, crop_size))
            vid_batch = np.concatenate(vid_batch).transpose((0, 4, 1, 2, 3))
        except:
            continue
        else:
            yield (vid_batch, one_hot_labels)

def create_h5py_dataset(txt_loc, formats=".mp4", db_name ="data_h5py"):
    num_lines = sum(1 for line in open(txt_loc))
    fi = FeatureIndexer(db_name, num_lines, 16*16)
    for k, i in tqdm(enumerate(open(txt_loc, "r"))):
        loc = i.split()
        label = loc[0].rsplit("/")[-2]
        vid_loc = loc[0]
        start_frame = int(loc[1])
        end_frame = int(loc[2])
        vid = imageio.get_reader(vid_loc, formats)
        imgs = [vid.get_data(k) for k in range(start_frame, end_frame+1)]
        imgs = preprocess_one_batch(imgs, True, (171, 128), False)
        imgs = np.concatenate([im[np.newaxis, :, :, :] for im in imgs])
        label = _labels.index(label)
        fi.add(imgs, label)
    fi.finish()

def create_h5py_dataset_2(txt_loc, formats=".mp4", db_name="data_h5py"):
    num_lines = sum(1 for line in open(txt_loc))
    fi = FeatureIndexer(db_name, num_lines, 64)
    for k, i in tqdm(enumerate(open(txt_loc, "r"))):
        vid_loc = i.split()[0]
        vid = imageio.get_reader(vid_loc, ".avi")
        nframes = vid.get_meta_data()["nframes"]
        label = vid_loc.rsplit("/")[-2]
        label = _labels.index(label)
        for fra in gen_frame(nframes, 16, 8):
            imgs = [vid.get_data(k) for k in fra]
            imgs = preprocess_one_batch(imgs, True, (171, 128), False)
            imgs = np.concatenate([im[np.newaxis, :, :, :] for im in imgs])
            imgs = imgs.transpose((3, 0, 1, 2))[np.newaxis, :, :, :, :]
            #print(imgs.shape)
            fi.add(imgs, label)
    fi.finish()


if __name__ == '__main__':
    # for i, k in enumerate(train_gen("txt_files/train_real.txt", 16, ".avi", True, (171, 128), True, (112, 112))):
    #     print(i, k[0].shape, k[1].shape)
    ## Write some test cases if possible

    #create_h5py_dataset("txt_files/train_real.txt", ".avi", "vid_data_h5py")
    create_h5py_dataset_2("txt_files/train.txt", ".avi", "vid_data_h5py")
