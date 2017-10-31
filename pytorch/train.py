""" train model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os

from torch_nets import *
from npy_creator import *
from data_loader import train_model
import pickle

train_file = "txt_files/train_real.txt"
valid_file = "txt_files/valid_real.txt"

# data_loaders = {"train": train_gen(train_file, 16, ".avi"),
#                 "val": train_gen(valid_file, 16, ".avi")}

data_loaders = {"train": train_file,
                 "val": valid_file}

num_classes = 101
use_gpu = True

print("[Load the model...]")
model_ft = Net_3d(num_classes)

print("[Using CrossEntropyLoss...]")
criterion = nn.CrossEntropyLoss()

if use_gpu:
    model_ft = model_ft.cuda()
    criterion = criterion.cuda()




print("[Using small learning rate with momentum...]")
# Observe that all parameters are being optimized
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)


print("[Creating Learning rate scheduler...]")
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

print("[Training the model begun ....]")
model_ft = train_model(model_ft, data_loaders, criterion, optimizer_ft, exp_lr_scheduler, batch_size=16,
                       num_epochs=10)


model_ft.save_dict('3d_model_weights.pt')

pickle.dumps(model_ft, open("model.pkl", "wb"))
