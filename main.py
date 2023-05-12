import itertools
import json
import os
import sys
from threading import Thread
from time import sleep

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

# minimum = 1000
# files = iou.get_wav_files(limit=7000)
# for file in files:
#     print('\n\n')
#     print('checking file ', file.get_file_name())
#     spec = torch.tensor(iou.get_spectrogram_from_waveform_in_db(file.waveform_data))
#     print('Spec timespan ', spec.shape[2])
#     print('Current minimum ', minimum)
#     if spec.shape[2] < minimum:
#         minimum = spec.shape[2]
#
# print(minimum)


def test_model_for_windows():
    dataset = data_manager_spectrogram.DataManagerSpectrograms(config).get_dataset_no_loader()
    model = networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)

    checkpoint = torch.load(
        os.path.join(config['exp_path'], config['exp_name_spec'], 'latest_checkpoint.pkl'),
        map_location=config['device'])
    model.load_state_dict(checkpoint['model_weights'])

    # enumerate ads a counter for each element in for loop ( like an i index )
    # this index corresponds to the batch being processed
    # enumerate returns ( index, obj ), and here object is ( spectrogram, label )

    for (spec, label) in dataset:
        spec = spec.cuda()
        label = label.cuda()

        windows = iou.spectrogram_windowing(spec, window_size=120, window_count=3).cuda()

        predictions = model(windows)

        predictions = predictions.detach().cpu().numpy()
        entropies = iou.compute_entropy(predictions)
        # predictions = torch.empty((3, ))


        # if self.config['windowing']:
        #     for i in range(0, len(spectrogram_batch)):
        #         windows = iou.spectrogram_windowing(spectrogram_batch[i]).cuda()
        #         emotion_label = emotion_prediction_batch[i]
        #         window_predictions = self.model(windows)
        #         correct_label_index = torch.argmax(emotion_label)
        #         correct_pred_column = window_predictions[:, correct_label_index.item()]
        #         best_window = torch.argmax(correct_pred_column)
        #         best_window_batch.append(best_window.item())
        #         best_window_predictions.append(window_predictions[best_window.item()])
        #
        #         self.log_file.write(
        #             f'\n\nBest window for {iou.get_labels[correct_label_index.item()]} is window {best_window.item()}/5'
        #             f'\nwith prediction: {window_predictions[best_window.item()]}\n')


test_model_for_windows()
