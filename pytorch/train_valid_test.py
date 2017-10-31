""" Creates train, valid and test txt files when given videos folder (which contains videos in folders according to class)
"""

import glob
import os
from sklearn.model_selection import train_test_split
import imageio
from npy_creator import gen_frame

files = glob.glob("../UCF-101/*")
print("Total files:", len(files))
txt_loc = "txt_files/"

if not os.path.exists(txt_loc):
    os.makedirs(txt_loc)


obj_names = [i.rsplit("/")[-1] for i in files]
names = open(txt_loc+"obj.names", "w")

for i in obj_names:
    names.write(i+"\n")

train = open(txt_loc+ 'train.txt', 'w')
valid = open(txt_loc+ 'valid.txt', 'w')
test  = open(txt_loc+ 'test.txt', 'w')

for c in files:
    k = glob.glob(c+"/*.avi")
    print("total files:", len(k))
    x_train, x_valid = train_test_split(k, test_size = 0.4, random_state = 2017)
    x_valid, x_test = train_test_split(x_valid, test_size = 0.5, random_state = 2017)
    for t in x_train:
        train.write(t+"\n")
    for v in x_valid:
        valid.write(v+"\n")
    for te in x_test:
        test.write(v+"\n")

print("[Saved all the files....]")
print("Building the training dataset")

train = open(txt_loc+"train_real.txt", "w")

for k , i in enumerate(open(txt_loc+"train.txt", "r")):
    vid_loc = i.split()[0]
    vid = imageio.get_reader(vid_loc, ".avi")
    nframes = vid.get_meta_data()["nframes"]
    print("sno: ", k, "vid_loc: ", vid_loc, "total_frames: ", nframes, " fps: ", vid.get_meta_data()["fps"] )

    for fra in gen_frame(nframes, 16, 8):
        train.write(vid_loc + ' ' + ' '.join([str(fra[0]), str(fra[-1])]) + "\n")



valid = open(txt_loc+"valid_real.txt", "w")

for k , i in enumerate(open(txt_loc+"valid.txt", "r")):
    vid_loc = i.split()[0]
    vid = imageio.get_reader(vid_loc, ".avi")
    nframes = vid.get_meta_data()["nframes"]
    print("sno: ", k, "vid_loc: ", vid_loc, "total_frames: ", nframes, " fps: ", vid.get_meta_data()["fps"] )

    for fra in gen_frame(nframes, 16, 8):
        valid.write(vid_loc + ' ' + ' '.join([str(fra[0]), str(fra[-1])]) + "\n")
