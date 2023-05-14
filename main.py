import json
import os

import numpy as np
from torch import cuda
from torch import load
from torch import nn
from torch import optim
from torchvision import models

import trainer as tr
from data import data_manager_spectrogram
from networks_files import networks, res_net
from util.ioUtil import spectrogram_windowing, compute_entropy, get_frame_from_video

config = json.load(open('config.json'))

device = "cuda" if cuda.is_available() else "cpu"
print(f"Using {device} device")


def run_training():
    model = networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
    train_dataloader, eval_dataloader = data_manager_spectrogram.DataManagerSpectrograms(
        config).get_train_eval_dataloaders_spectrograms()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=7)

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

    checkpoint = load(
        os.path.join(config['exp_path'], config['exp_name_spec'], 'latest_checkpoint.pkl'),
        map_location=config['device'])
    model.load_state_dict(checkpoint['model_weights'])

    for (spec, label, spec_path) in dataset:
        spec = spec.cuda()
        label = label.cuda()

        windows, windows_indexes = spectrogram_windowing(spec, window_size=120, window_count=3)
        windows = windows.cuda()

        predictions, terminal_layers = model(windows)

        predictions = predictions.detach().cpu().numpy()
        entropies = compute_entropy(predictions)

        best_window_index = np.argmin(entropies)

        videos_dir = config['video_dir_path']
        file_name = spec_path.split('/')[-1]
        video_path = os.path.join(videos_dir, file_name)

        start, end = windows_indexes[best_window_index]

        frame = get_frame_from_video(video_path=video_path, start_index=start, end_index=end,
                                     spectrogram_length=spec.shape[2], plot=True)
        spec_terminal_layer = terminal_layers[best_window_index]


# test_model_for_windows()


# TODO steps
# video enters 👍
# split data in video and audio 👍
# preprocess audio data 👍
# send spectrogram to spec model 👍
# preprocess video data 👍
# use spec model terminal layer and video data ( currently 1 image ) to feed pretrained model
# see the results

# testing pretrained model


def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.

    model = models.resnet152(pretrained=use_pretrained)
    set_parameter_requires_grad(model, feature_extract)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 224

    return model, input_size


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


num_classes = 6
feature_extract = True
# Initialize the model for this run
model, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model)

# TODO
# add trainer class for video
# add train methods
# build the dataloader
# feed the image to the pretrained model
