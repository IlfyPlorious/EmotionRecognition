import os

import numpy as np
import torch
from torch import load
from torch import nn
from torchvision import models
from torchvision.transforms import Lambda

from data.audio_video_dataset import AudioVideoDataset
from networks_files.networks import ResNet
from networks_files.res_net import BasicBlock
from util import ioUtil


class DataManager:
    '''
    Final datamanager using spectrogram model and pretrained model to transform dataset
    in order to obtain final avgpool layers to be concatenated
    '''

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
        # for actor in os.listdir(self.config['spectrogram_dir']):
        #     self.actor_dirs.append(os.path.join(self.config['spectrogram_dir'], actor))
        #     self.actor_dirs_count += 1

        for actor in range(1001, 1009):
            self.actor_dirs.append(os.path.join(self.config['spectrogram_dir'], str(actor)))
            self.actor_dirs_count += 1

        self.vid_model = self.initialize_pretrained_video_model()
        self.spec_model = self.initialize_spectrogram_model()

    def initialize_spectrogram_model(self, device='cuda'):
        model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)

        checkpoint = load(
            os.path.join(self.config['exp_path'], self.config['exp_name_spec'], 'latest_checkpoint.pkl'),
            map_location=self.config['device'])
        model.load_state_dict(checkpoint['model_weights'])

        return model

    def initialize_pretrained_video_model(self, num_classes=6, feature_extract=True, use_pretrained=True,
                                          device='cuda'):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.

        model = models.resnet50(pretrained=use_pretrained).to(device)
        self.set_parameter_requires_grad(model, feature_extract)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

        return model

    @staticmethod
    def set_parameter_requires_grad(model, feature_extracting=True):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def get_dataloader(self):
        dataset = AudioVideoDataset(window_count=3, window_size=120, actor_dirs=self.actor_dirs,
                                    transform=self.transform,
                                    transform_target=self.transform_target, config=self.config,
                                    spec_model=self.spec_model, vid_model=self.vid_model)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            pin_memory=False
        )
        return dataloader

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

        train_dataset = AudioVideoDataset(window_count=3, window_size=120, actor_dirs=train_dirs,
                                          transform=self.transform,
                                          transform_target=self.transform_target, config=self.config,
                                          spec_model=self.spec_model, vid_model=self.vid_model)
        train_dataset_size = len(train_dataset)

        eval_dataset = AudioVideoDataset(window_count=3, window_size=120, actor_dirs=eval_dirs,
                                         transform=self.transform,
                                         transform_target=self.transform_target, config=self.config,
                                         spec_model=self.spec_model, vid_model=self.vid_model)
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
                                                   drop_last=True,
                                                   pin_memory=False)

        validation_loader = torch.utils.data.DataLoader(dataset=eval_dataset,
                                                        batch_size=self.config['batch_size'],
                                                        sampler=valid_sampler,
                                                        drop_last=True,
                                                        pin_memory=False)

        return train_loader, validation_loader
