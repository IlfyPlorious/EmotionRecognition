import os.path

import numpy as np
import torch
from torch.utils.data import Dataset

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
        with torch.no_grad():
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

        sorted_frame_numbers = sorted(frames)

        randomizer_start_end = np.random.randint(low=0, high=4)
        randomizer_mid = np.random.randint(low=-2, high=2)

        start_frame_name = None
        end_frame_name = None
        mid_frame_name = None

        try:
            start_frame_name = [
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * start // spectrogram_length) + randomizer_start_end],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * start // spectrogram_length) + randomizer_start_end + 1],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * start // spectrogram_length) + randomizer_start_end + 2]
            ]
            end_frame_name = [
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // spectrogram_length) - randomizer_start_end - 1],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // spectrogram_length) - randomizer_start_end - 2],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // spectrogram_length) - randomizer_start_end - 3]
            ]
            mid_frame_name = [
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // (spectrogram_length * 2)) + randomizer_mid - 1],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // (spectrogram_length * 2)) + randomizer_mid],
                sorted_frame_numbers[
                    (len(sorted_frame_numbers) * end // (spectrogram_length * 2)) + randomizer_mid + 1]
            ]
        except Exception as e:
            print(e)
            print(f'Problem with {vid_name}')
            print(f'Length of sorted_frame_numbers: {len(sorted_frame_numbers)}')
            print(f'randomizer1: {randomizer_start_end}')
            print(f'randomizer2: {randomizer_mid}')
            print(f'start index: {(len(sorted_frame_numbers) * start // spectrogram_length) + randomizer_start_end}')
            print(f'end index: {(len(sorted_frame_numbers) * end // spectrogram_length) - randomizer_start_end - 1}')
            print(
                f'mid index: {(len(sorted_frame_numbers) * (start + end) // (spectrogram_length * 2)) + randomizer_mid}')

        try:
            start_stack = []
            mid_stack = []
            end_stack = []
            for frame_name in start_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
                start_stack.append(frame)

            for frame_name in mid_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
                mid_stack.append(frame)

            for frame_name in end_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
                end_stack.append(frame)

            start_stack = torch.stack(start_stack)
            mid_stack = torch.stack(mid_stack)
            end_stack = torch.stack(end_stack)

            frame_image_features = torch.cat((
                start_stack,
                mid_stack,
                end_stack
            ), dim=1)

            frame_image_features = torch.stack([frame_image_features])
        except Exception as e:
            print(e)
            print(f'Problems with {vid_name}')
            print(f'Frame names for {vid_name}: {start_frame_name}, {mid_frame_name}, {end_frame_name}')
            print(f'Frames for {vid_name}: {frame_numbers}')
            print(f'Start index: {start_frame_name}')
            print(f'Mid index: {mid_frame_name}')
            print(f'End index: {end_frame_name}')
            print(f'Spec path {spec_path}')

        file_name = spec_path.split('/')[-1]
        return spec_terminal_layer, frame_image_features, label, file_name
