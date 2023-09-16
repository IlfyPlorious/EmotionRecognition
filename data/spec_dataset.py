import os.path

import numpy as np
import torch
from torch.utils.data import Dataset
from util import ioUtil as iou

"""

-------------------  SPECTROGRAM DATASET -------------------------

"""


class SpectrogramsDataset(Dataset):
    def __init__(self, actor_dirs, window_count=3, window_size=120, no_windowing=False, transform=None,
                 transform_target=None, config=None, return_names=False):
        self.transform = transform
        self.target_transform = transform_target
        self.actor_dirs = actor_dirs
        self.window_count = window_count
        self.window_size = window_size
        self.no_windowing = no_windowing
        self.config = config
        self.return_names = return_names

    def __len__(self):
        length = 0
        for actor_dir in self.actor_dirs:
            for _ in os.listdir(actor_dir):
                length += 1

        if self.no_windowing:
            return length

        return length * self.window_count

    def get_spectrogram_paths_list(self):
        spectrogram_paths = []

        for actor_dir in self.actor_dirs:
            for spec_name in os.listdir(actor_dir):
                spectrogram_paths.append((os.path.join(actor_dir, spec_name), spec_name.split('_')[2]))

        return spectrogram_paths

    def __getitem__(self, idx):
        spec_idx = idx // self.window_count
        if self.no_windowing:
            spec_idx = idx

        spectrograms = self.get_spectrogram_paths_list()
        spectrogram_path = spectrograms[spec_idx]
        spec_path = spectrogram_path[0]  # element in tuple at index 0 is the spectrogram path
        label = spectrogram_path[1]  # element in tuple at index 1 is the label
        spec = np.load(spec_path)
        spec = torch.tensor(spec)

        if self.transform:
            spec = self.transform(spec)
        if self.target_transform:
            label = self.target_transform(label)

        if self.no_windowing:
            if self.return_names:
                return spec, label, spec_path
            else:
                return spec, label

        if self.config['random_windowing']:
            windows = iou.random_spectrogram_windowing(spec, window_size=self.window_size,
                                                       window_count=self.window_count,
                                                       plot=False)

            window = windows[idx % self.window_count]

            return window, label
        else:
            windows = iou.spectrogram_windowing(spec, window_size=self.window_size, window_count=self.window_count,
                                                plot=False)

            window = windows[idx % self.window_count]

            return window, label