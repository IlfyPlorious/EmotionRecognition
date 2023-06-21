import torch.nn.functional as F
from torch import nn


class EmotionsNetwork2LinLayers(nn.Module):
    def __init__(self):
        super(EmotionsNetwork2LinLayers, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(369 * 496 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 6)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class EmotionsNetwork2Conv2Layers(nn.Module):
    def __init__(self):
        super(EmotionsNetwork2Conv2Layers, self).__init__()
        # 4 channels because spectrogram tensors shape is [4, 369, 496] which has 4 channels
        self.conv1 = nn.Conv2d(4, 8, 30)
        self.conv2 = nn.Conv2d(8, 16, 30)
        self.fc1 = nn.Linear(4 * 8 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        # Max pooling over a (6, 6) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (6, 6))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 6)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EmotionsNetworkV3(nn.Module):
    def __init__(self):
        super(EmotionsNetworkV3, self).__init__()
        # 4 channels because spectrogram tensors shape is [4, 369, 496] which has 4 channels
        self.conv1 = nn.Conv2d(1, 8, 10)
        self.conv2 = nn.Conv2d(8, 16, 10)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        # Max pooling over a (6, 6) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (6, 6))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 6)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
