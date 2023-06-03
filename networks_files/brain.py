import os

import numpy as np
import torch
from torch import nn

from networks_files.hook import Hook
from util.ioUtil import spectrogram_windowing, compute_entropy, get_frame_from_video


class Brain(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(Brain, self).__init__()

        self.fc1 = nn.Linear(2048 * 3, 2048)
        self.fc2 = nn.Linear(2048 + 512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = self.fc1(x)
        x = self.fc2(torch.cat((x, y), dim=1))
        x = self.fc3(x)
        x = self.softmax(x)

        return x


class Brain2(nn.Module):
    def __init__(self, num_classes: int = 6):
        super(Brain2, self).__init__()

        self.conv1 = nn.Conv2d(1, 1, kernel_size=(3, 1))
        self.fc1 = nn.Linear(2048 * 3, 2048)
        self.fc2 = nn.Linear(2048 + 512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(torch.cat((x, y), dim=1))
        x = self.fc3(x)
        x = self.softmax(x)

        return x
