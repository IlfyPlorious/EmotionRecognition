import os.path

import numpy as np
import torch
from torch.utils.data import Dataset

from networks_files.hook import Hook
from util.ioUtil import spectrogram_windowing, compute_entropy, get_frame_from_video


class AudioVideoDataset(Dataset):
    def __init__(self, actor_dirs, window_count=3, window_size=120, transform=None,
                 transform_target=None, config=None, spec_model=None, vid_model=None):
        self.transform = transform
        self.target_transform = transform_target
        self.actor_dirs = actor_dirs
        self.window_count = window_count
        self.window_size = window_size
        self.config = config

        self.spec_model = spec_model.to('cuda')
        self.vid_model = vid_model.to('cuda')
        self.hook = Hook()

    def __len__(self):
        length = 0
        for actor_dir in self.actor_dirs:
            for _ in os.listdir(actor_dir):
                length += 1

        return length

    def get_spectrogram_paths_list(self):
        spectrogram_paths = []

        for actor_dir in self.actor_dirs:
            for spec_name in os.listdir(actor_dir):
                spectrogram_paths.append((os.path.join(actor_dir, spec_name), spec_name.split('_')[2]))

        return spectrogram_paths

    def __getitem__(self, idx):
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

        windows, windows_indexes = spectrogram_windowing(spec, window_size=self.window_size,
                                                         window_count=self.window_count)
        windows = windows.cuda()
        predictions, terminal_layers = self.spec_model(windows)

        predictions = predictions.detach().cpu().numpy()
        entropies = compute_entropy(predictions)

        best_window_index = np.argmin(entropies)
        videos_dir = self.config['video_dir_path']
        file_name = spec_path.split('/')[-1]
        video_path = os.path.join(videos_dir, file_name)

        start, end = windows_indexes[best_window_index]
        if end == 0:
            pass

        frame = get_frame_from_video(video_path=video_path, start_index=start, end_index=end,
                                     spectrogram_length=spec.shape[2])
        frame = torch.tensor(frame).to('cuda')

        spec_terminal_layer = terminal_layers[best_window_index]

        layer = self.vid_model.get_submodule('avgpool')
        handle = layer.register_forward_hook(self.hook)

        if self.transform:
            frame = self.transform(frame).expand(1, -1, -1, -1).to('cuda')

        _ = self.vid_model(frame)

        video_terminal_layer = self.hook.outputs[0].squeeze()
        self.hook.clear()

        features = torch.cat((spec_terminal_layer, video_terminal_layer))

        return features, label
