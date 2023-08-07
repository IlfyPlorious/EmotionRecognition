import os

import numpy as np
import torch
from torchvision.transforms import Lambda

from data.spec_dataset import SpectrogramsDataset
from util import ioUtil


class DataManagerSpectrograms:
    """Manager class for data loaders.

Config argument is a dictionary that contains the following:

spectrogram_dir -> path_to_spectograms_dir

video_data -> path_to_frames_dir

batch_size -> size of the batch

train_epochs -> epochs count

device -> cuda if gpu else cpu

train_split -> division part for train size = dataset // train_split

valid_split -> division part for validation size = dataset // valid_split

"""

    def __init__(self, config):
        self.config = config
        self.transform = Lambda(lambda tensor: ioUtil.map_tensor_to_0_1(tensor))
        self.transform_target = Lambda(lambda label:
                                       torch.zeros(len(ioUtil.labels.values()),
                                                   dtype=torch.float)
                                       .scatter_(dim=0,
                                                 index=torch.tensor(
                                                     ioUtil.labels.get(
                                                         label)),
                                                 value=1))
        self.actor_dirs = []
        self.actor_dirs_count = 0
        for actor in os.listdir(self.config['spectrogram_dir']):
            self.actor_dirs.append(os.path.join(self.config['spectrogram_dir'], actor))
            self.actor_dirs_count += 1

    def get_dataloader_spectrograms(self):
        dataset = SpectrogramsDataset(window_count=3, window_size=120, actor_dirs=self.actor_dirs,
                                      transform=self.transform,
                                      transform_target=self.transform_target, config=self.config)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=True if self.config['device'] == 'cuda' else False
        )
        return dataloader

    def get_dataset_no_loader(self):
        dataset = SpectrogramsDataset(actor_dirs=self.actor_dirs, no_windowing=True,
                                      transform=self.transform,
                                      transform_target=self.transform_target, config=self.config, return_names=True)

        return dataset

    def get_train_eval_dataloaders_spectrograms(self):
        np.random.seed(707)

        ## SPLIT DATASET

        ## 0 < train_split < 1 ; which represents the percentage for train data

        train_split = self.config['train_split']
        train_size = int(self.actor_dirs_count * train_split)
        validation_size = self.actor_dirs_count - train_size

        train_dirs = self.actor_dirs[:train_size]
        temp = int(train_size + validation_size)
        eval_dirs = self.actor_dirs[train_size:temp]

        ## DATASETS SHUFFLE ##
        np.random.shuffle(train_dirs)
        np.random.shuffle(eval_dirs)

        train_dataset = SpectrogramsDataset(window_count=3, window_size=120, actor_dirs=train_dirs,
                                            transform=self.transform,
                                            transform_target=self.transform_target, config=self.config)
        train_dataset_size = len(train_dataset)

        eval_dataset = SpectrogramsDataset(window_count=3, window_size=120, actor_dirs=eval_dirs,
                                           transform=self.transform,
                                           transform_target=self.transform_target, config=self.config)
        eval_dataset_size = len(eval_dataset)

        ## INDICES SHUFFLE ##

        train_indices = list(range(train_dataset_size))
        np.random.shuffle(train_indices)
        eval_indices = list(range(eval_dataset_size))
        np.random.shuffle(eval_indices)

        ## These samplers shuffle the indices which will be used to extract data from dataset
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(eval_indices)

        ## DATA LOADER ##

        ## Obj representing a list of tuples with format ( spectrogram , label )
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.config['batch_size'],
                                                   sampler=train_sampler,
                                                   pin_memory=True if self.config['device'] == 'cuda' else False)

        validation_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        pin_memory=True if self.config['device'] == 'cuda' else False)

        return train_loader, validation_loader
