import itertools
import json
import os
import sys
from threading import Thread
from time import sleep

import cv2
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from data.base_dataset import SpectrogramsDataset
from util import ioUtil as iou

import trainer as tr
from data import data_manager_spectrogram
from networks_files import networks, res_net

config = json.load(open('config.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


def run_training():
    model = networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
    train_dataloader, eval_dataloader = data_manager_spectrogram.DataManagerSpectrograms(
        config).get_train_eval_dataloaders_spectrograms()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=7)

    trainer = tr.TrainerSpectrogram(model=model, train_dataloader=train_dataloader,
                                    eval_dataloader=eval_dataloader,
                                    loss_fn=nn.CrossEntropyLoss(), criterion=None, optimizer=optimizer,
                                    scheduler=scheduler,
                                    config=config)

    trainer.run()


# run_training()


def test_model_for_windows():
    dataset = data_manager_spectrogram.DataManagerSpectrograms(config).get_dataset_no_loader()
    model = networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)

    checkpoint = torch.load(
        os.path.join(config['exp_path'], config['exp_name_spec'], 'latest_checkpoint.pkl'),
        map_location=config['device'])
    model.load_state_dict(checkpoint['model_weights'])

    for (spec, label) in dataset:
        spec = spec.cuda()
        label = label.cuda()

        windows = iou.spectrogram_windowing(spec, window_size=120, window_count=5).cuda()

        predictions, terminal_layers = model(windows)

        predictions = predictions.detach().cpu().numpy()
        entropies = iou.compute_entropy(predictions)


# test_model_for_windows()


# video_dir_path = config['video_dir_path']
#
# file = os.listdir(video_dir_path)[10]
# print(file)
# video_file = os.path.join(video_dir_path, file)

# read specific frame
# video_capture = cv2.VideoCapture(video_file)
# frame_number = 24
# video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
# res, frame = video_capture.read()


# TODO steps
# video enters
# split data in video and audio
# preprocess audio data
# send spectrogram to spec model
# preprocess video data
# use spec model terminal layer and video data ( currently 1 image ) to feed pretrained model
# see the results
