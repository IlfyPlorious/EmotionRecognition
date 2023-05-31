import os.path

import numpy as np
import torch
from torch.utils.data import Dataset

from networks_files.hook import Hook
from util.ioUtil import spectrogram_windowing, compute_entropy, spectrogram_name_splitter


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

        ## ---------- MANAGE SPECTROGRAM --------- ##
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

        start, end = windows_indexes[best_window_index]
        spec_terminal_layer = terminal_layers[best_window_index]

        ## ---------- MANAGE VIDEO FRAME --------- ##

        actor, line, emotion, intensity, _ = spectrogram_name_splitter(spec_path.split('/')[-1])
        image_data_dir = self.config['video_data']
        image_actor_dir = os.path.join(image_data_dir, actor)
        vid_name = f'{actor}_{line}_{emotion}_{intensity}'

        vids = os.listdir(image_actor_dir)
        frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))

        i = 0
        intensities = ['LO', 'MD', 'HI']

        while len(frames) == 0 and i < 3:
            vid_name = f'{actor}_{line}_{emotion}_{intensities[i]}'
            frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))
            i += 0

        spectrogram_length = spec.shape[2]
        frame_numbers = list(map(lambda frame: int(frame.split('_')[-1].split('.')[0]), frames))
        max_frame_number = np.amax(frame_numbers)

        start_index_frames = max_frame_number * start // spectrogram_length
        end_index_frames = max_frame_number * end // spectrogram_length

        if start_index_frames == 0:
            start_index_frames = np.amin(frame_numbers)

        frames_names = []

        for i in range(0, len(frame_numbers)):
            if frame_numbers[i] == start_index_frames:
                frames_names.append(frames[i])
            if frame_numbers[i] == (start_index_frames + end_index_frames) // 2:
                frames_names.append(frames[i])
            if frame_numbers[i] == end_index_frames - 1:
                frames_names.append(frames[i])

        # todo add random +- 2 for midle +- 4 for edges
        frame_image_features = torch.cat((
            torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[0]))),
            torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[1]))),
            torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[2]))))
        )
        frame_image_features = torch.tensor(frame_image_features).cuda()
        # frame_image = np.transpose(frame_image, (2, 0, 1))
        # frame_image = torch.tensor([frame_image]).float().cuda()
        #
        # layer = self.vid_model.get_submodule('avgpool')
        # handle = layer.register_forward_hook(self.hook)
        #
        # _ = self.vid_model(frame_image)
        #
        # video_terminal_layer = self.hook.outputs[0].squeeze()
        # self.hook.clear()
        #

        file_name = spec_path.split('/')[-1]
        return spec_terminal_layer, frame_image_features, label, file_name
