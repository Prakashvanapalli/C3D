""" Functions for data loader
"""
import time
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import os
import sys
from npy_creator import *

sys.stdout.write('\r')
# the exact output you're looking for:

sys.stdout.flush()

use_gpu = True

def train_model(model, dataloders,  criterion, optimizer, scheduler, batch_size=32, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        data_loaders = {"train": train_gen(dataloders["train"], batch_size, ".avi"),
                         "val": train_gen(dataloders["val"], batch_size, ".avi")}

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for k, data in enumerate(data_loaders[phase]):
                sys.stdout.write('\r')
                # get the inputs
                inputs, labels = data
                inputs, labels = torch.from_numpy(inputs/255.0).float(), torch.from_numpy(labels).float().long()

                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

                if k:
                    sys.stdout.write("'Iteration:' {} 'Loss:' {} 'Acc' {}".format( k, running_loss/(k), running_corrects/(k*16.0)))
                sys.stdout.flush()

            epoch_loss = running_loss / (k)
            epoch_acc = running_corrects / (k*16)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
