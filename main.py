import json
import os

import numpy as np
import torch
from torch import cuda
from torch import load
from torch import nn
from torch import optim

import trainer as tr
from brain_trainer import BrainTrainer
from data import data_manager_spectrogram
from data.data_manager import DataManager
from networks_files import networks, res_net
from networks_files.brain import Brain, Brain2
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


# def initialize_model(num_classes, feature_extract, use_pretrained=True):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#
#     model = models.resnet50(pretrained=use_pretrained)
#     set_parameter_requires_grad(model, feature_extract)
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, num_classes)
#     input_size = 224
#
#     return model, input_size
#
#
# def set_parameter_requires_grad(model, feature_extracting):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
#

# num_classes = 6
# feature_extract = True
# # Initialize the model for this run
# vid_model, input_size = initialize_model(num_classes, feature_extract, use_pretrained=True)
# # Print the model we just instantiated
# # print(model.get_submodule(''))
# hook = Hook()
# layer = vid_model.get_submodule('avgpool')
# handle = layer.register_forward_hook(hook)
#
# x = vid_model(torch.empty((1, 3, 120, 120)))
#
# ten = hook.outputs[0].clone().squeeze().cuda()
#
# dataset = data_manager_spectrogram.DataManagerSpectrograms(config).get_dataset_no_loader()
# model = networks.ResNet(block=res_net.BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
#
# checkpoint = load(
#     os.path.join(config['exp_path'], config['exp_name_spec'], 'latest_checkpoint.pkl'),
#     map_location=config['device'])
# model.load_state_dict(checkpoint['model_weights'])
#
# spec, label, spec_path = dataset[0]
#
# spec = spec.cuda()
# label = label.cuda()
#
# windows, windows_indexes = spectrogram_windowing(spec, window_size=120, window_count=3)
# windows = windows.cuda()
#
# predictions, terminal_layers = model(windows)
#
# predictions = predictions.detach().cpu().numpy()
# entropies = compute_entropy(predictions)
#
# best_window_index = np.argmin(entropies)
#
# videos_dir = config['video_dir_path']
# file_name = spec_path.split('/')[-1]
# video_path = os.path.join(videos_dir, file_name)
#
# start, end = windows_indexes[best_window_index]
#
# frame = get_frame_from_video(video_path=video_path, start_index=start, end_index=end,
#                              spectrogram_length=spec.shape[2])
# spec_terminal_layer = terminal_layers[best_window_index].cuda()
#
#
# x = torch.cat((spec_terminal_layer, ten))
#
# print(spec_terminal_layer.shape)
# print(ten.shape)
# print(x.shape)

# TODO
# add trainer class for video
# add train methods
# build the dataloader
# feed the image to the pretrained model

# video_dir = config['video_dir_path']
# files = os.listdir(video_dir)
# file1 = os.path.join(video_dir, files[0])
# file2 = os.path.join(video_dir, files[1])
# video_capture1 = cv2.VideoCapture(file1)
# video_capture2 = cv2.VideoCapture(file2)
# total_frames1 = video_capture1.get(cv2.CAP_PROP_FRAME_COUNT)
# total_frames2 = video_capture2.get(cv2.CAP_PROP_FRAME_COUNT)
#
# print(total_frames1)
# print(total_frames2)
#
# frame_number = np.random.randint(low=0, high=50)
# video_capture1.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
# video_capture2.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
# _, frame1 = video_capture1.read()
# _, frame2 = video_capture2.read()
# video_capture1.release()
# video_capture2.release()
#
# plt.figure(), plt.imshow(frame1)
# plt.figure(), plt.imshow(frame2)
# plt.show()


# steps
# build a new network
# image enters the pretrained model
# gives out the las layer
# concatenate last layer from spec with last layer from video
# add 2 Linear layers 512 then 6
# see the output


def run_brain_training():
    # model = Brain().to(device)
    model = Brain2().to(device)
    train_dataloader, eval_dataloader = DataManager(config=config).get_train_eval_dataloaders_spectrograms()
    optimizer = optim.SGD(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, total_iters=config['train_epochs'] * 0.5)

    trainer = BrainTrainer(model=model, train_dataloader=train_dataloader,
                           eval_dataloader=eval_dataloader,
                           loss_fn=nn.CrossEntropyLoss(), criterion=None, optimizer=optimizer,
                           scheduler=scheduler,
                           config=config)

    trainer.run()


run_brain_training()

# from util.ioUtil import map_tensor_to_0_1
#
# from networks_files.networks import ResNet
# from networks_files.res_net import BasicBlock
#
#
# def initialize_spectrogram_model(device='cuda'):
#     model = ResNet(block=BasicBlock, layers=[1, 1, 1, 1], num_classes=6).to(device)
#
#     checkpoint = load(
#         os.path.join(config['exp_path'], config['exp_name_spec'], 'latest_checkpoint.pkl'),
#         map_location=config['device'])
#     model.load_state_dict(checkpoint['model_weights'])
#
#     return model
#
#
# spec_model = initialize_spectrogram_model()
# # run_brain_training()
# spec_path = '/home/dragos/Desktop/Facultate/Licenta/Emotions/SpectrogramData/1076/1076_ITH_HAPPY_UNSPECIFIED.npy'
# spec = np.load(spec_path)
# spec = torch.tensor(spec)
# spec = map_tensor_to_0_1(spec)
#
# windows, windows_indexes = spectrogram_windowing(spec, window_size=120,
#                                                  window_count=3)
# windows = windows.cuda()
# predictions, terminal_layers = spec_model(windows)
#
# predictions = predictions.detach().cpu().numpy()
# entropies = compute_entropy(predictions)
#
# best_window_index = np.argmin(entropies)
# videos_dir = config['video_dir_path']
# file_name = spec_path.split('/')[-1]
# video_path = os.path.join(videos_dir, file_name)
# frame = get_frame_from_video(video_path=video_path, start_index=0, end_index=20,
#                              spectrogram_length=120)

# write_video_frames_as_npy()

## write pretrained resnet features
# def initialize_pretrained_video_model(num_classes=6, feature_extract=True, use_pretrained=True,
#                                       device='cuda'):
#     # Initialize these variables which will be set in this if statement. Each of these
#     #   variables is model specific.
#
#     model = models.resnet50(pretrained=use_pretrained).to(device)
#     set_parameter_requires_grad(model, feature_extract)
#     num_features = model.fc.in_features
#     model.fc = nn.Linear(num_features, num_classes)
#
#     return model
#
#
# def set_parameter_requires_grad(model, feature_extracting=True):
#     if feature_extracting:
#         for param in model.parameters():
#             param.requires_grad = False
#
#
# model = initialize_pretrained_video_model()
#
# start = 1091
# write_pretrained_model_features_for_video(model, start=start, end=1092)


## check faulty farmes
# image_data_dir = config['video_data']
# image_actor_dir = os.path.join(image_data_dir, '1033')
# vid_name = '1033_IWL_HAP_XX'
# actor, line, emotion, _ = vid_name.split('_')
# vids = os.listdir(image_actor_dir)
# frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))
#
# i = 0
# intensities = ['LO', 'MD', 'HI']
#
# while len(frames) == 0 and i < 3:
#     vid_name = f'{actor}_{line}_{emotion}_{intensities[i]}'
#     frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))
#     i += 0
#
# spec = np.load('SpectrogramData/1033/1033_MTI_HAPPY_UNSPECIFIED.npy')
# spectrogram_length = spec.shape[2]
# start = 0
# end = 120
# frame_numbers = list(map(lambda frame: int(frame.split('_')[-1].split('.')[0]), frames))
# max_frame_number = np.amax(frame_numbers)
#
# start_index_frames = max_frame_number * start // spectrogram_length
# end_index_frames = max_frame_number * end // spectrogram_length
#
# if start_index_frames == 0:
#     start_index_frames = np.amin(frame_numbers)
#
# frames_names = []
#
# for i in range(0, len(frame_numbers)):
#     if frame_numbers[i] == start_index_frames:
#         frames_names.append(frames[i])
#     if frame_numbers[i] == (start_index_frames + end_index_frames) // 2:
#         frames_names.append(frames[i])
#     if frame_numbers[i] == end_index_frames - 1:
#         frames_names.append(frames[i])
#
# # todo add random +- 2 for midle +- 4 for edges
# try:
#     frame_image_features = torch.cat((
#         torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[0]))),
#         torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[1]))),
#         torch.tensor(np.load(os.path.join(image_actor_dir, frames_names[2]))))
#     )
# except Exception as e:
#     print(e)
#     print(f'Problems with {vid_name}')
#
# frame_image_features = torch.tensor(frame_image_features).cuda()

