""" Building a pytorch model for training Deep Neural Networks
"""

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net_3d(nn.Module):

    def __init__(self, n_classes):
        super(Net_3d, self).__init__()
        self.n_classes = n_classes
        self.conv1a = nn.Conv3d(3, 64, 3, padding = 1)
        self.conv2a = nn.Conv3d(64, 128, 3, padding = 1)
        self.conv3a = nn.Conv3d(128, 256, 3, padding = 1)
        self.conv3b = nn.Conv3d(256, 256, 3, padding = 1)
        self.conv4a = nn.Conv3d(256, 512, 3, padding = 1)
        self.conv4b = nn.Conv3d(512, 512, 3, padding = 1)
        self.conv5a = nn.Conv3d(512, 512, 3, padding = 1)
        self.conv5b = nn.Conv3d(512, 512, 3, padding = 1)

        self.fc1 = nn.Linear(4608, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.n_classes)

    def forward(self, x):
        x = F.max_pool3d(F.relu(self.conv1a(x)), (1, 2, 2))
        x = F.max_pool3d(F.relu(self.conv2a(x)), 2)
        x = F.relu(self.conv3a(x))
        x = F.max_pool3d(F.relu(self.conv3b(x)), 2)
        x = F.relu(self.conv4a(x))
        x = F.max_pool3d(F.relu(self.conv4b(x)), 2)
        x = F.relu(self.conv5a(x))
        x = F.max_pool3d(F.relu(self.conv5b(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
