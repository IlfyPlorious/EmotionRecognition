import os

import numpy as np
import torch
from torch import nn

from networks_files.hook import Hook
from util.ioUtil import spectrogram_windowing, compute_entropy, get_frame_from_video


class Brain(nn.Module):
    def __init__(self, spec_model, video_model, config, num_classes: int = 6, ):
        super(Brain, self).__init__()

        self.spec_model = spec_model
        self.video_model = video_model
        self.config = config
        self.hook = Hook()
        self.fc1 = nn.Linear(2560, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x
