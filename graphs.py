import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from data.data_manager import DataManager
from data.data_manager_spectrogram import DataManagerSpectrograms
from networks_files.brain import Brain
from networks_files.networks import SpectrogramBrain
from networks_files.res_net import BasicBlock

config = json.load(open('config.json'))
device = "cuda" if torch.cuda.is_available() else "cpu"

brain_checkpoint = torch.load(
    os.path.join(config['final_exp_path'], config['final_exp_name'],
                 'latest_checkpoint.pkl'),
    map_location=config['device'])

spec_checkpoint = torch.load(
    os.path.join(config['exp_path'], config['exp_name_spec'],
                 'latest_checkpoint.pkl'),
    map_location=config['device']
)

brain_model = Brain().to(device)
train_dataloader_brain, eval_dataloader_brain = DataManager(config=config).get_train_eval_dataloaders_audiovideo()
brain_model.load_state_dict(brain_checkpoint['model_weights'])

spec_model = SpectrogramBrain(block=BasicBlock, layers=[2, 2, 3, 2], num_classes=6).to(device)
train_dataloader_spec, eval_dataloader_spec = DataManagerSpectrograms(
    config=config).get_train_eval_dataloaders_spectrograms()
spec_model.load_state_dict(spec_checkpoint['model_weights'])


def get_conf_mat_brain():
    conf_mat = np.zeros((6, 6))
    with torch.no_grad():
        for batch, (spec_features_batch, vid_features_batch, labels_batch, file_names) in enumerate(
                eval_dataloader_brain, start=1):
            vid_features_batch = vid_features_batch.to(device)
            spec_features_batch = spec_features_batch.to(device)
            labels_batch = labels_batch.to(device)
            output = brain_model(vid_features_batch, spec_features_batch).cuda()

            indexes = []
            for index, name in enumerate(file_names):
                if 'DIS' in name or 'dis' in name:
                    indexes.append(index)

            if len(indexes) != 0:
                pass

            predictions = output.argmax(axis=1)
            expected = labels_batch.argmax(axis=1)

            predictions = predictions.detach().cpu().numpy()
            expected = expected.detach().cpu().numpy()

            conf_mat = np.add(conf_mat, confusion_matrix(predictions, expected, labels=[0, 1, 2, 3, 4, 5]))

    return conf_mat


def get_conf_mat_spec():
    conf_mat = np.zeros((6, 6))
    with torch.no_grad():
        for batch, (spectrogram_batch, emotion_prediction_batch) in enumerate(
                eval_dataloader_spec, start=0):
            spectrogram_batch = spectrogram_batch.to(device)
            emotion_prediction_batch = emotion_prediction_batch.to(device)

            output, _ = spec_model(spectrogram_batch)
            output = output.to(device)

            predictions = output.argmax(axis=1)
            expected = emotion_prediction_batch.argmax(axis=1)

            predictions = predictions.detach().cpu().numpy()
            expected = expected.detach().cpu().numpy()

            conf_mat = np.add(conf_mat, confusion_matrix(predictions, expected, labels=[0, 1, 2, 3, 4, 5]))

    return conf_mat


print('Generating confusion matrix...')
# conf_mat = get_conf_mat_spec()
conf_mat = get_conf_mat_brain()
h, w = conf_mat.shape
for i in range(0, h):
    sum = np.sum(conf_mat[i])
    if sum != 0:
        conf_mat[i] = np.multiply(np.divide(conf_mat[i], sum), 100)

print("Done!")
sns.heatmap(conf_mat,
            annot=True)
plt.ylabel('Predicții', fontsize=13)
plt.xlabel('Etichete', fontsize=13)
plt.title('Matrice de confuzie', fontsize=17)
plt.show()

# log = 'Logs/spec_2023-06-23.csv'
log = 'Logs/brain_new_2023-06-24.csv'

test_accuracies = []
train_accuracies = []
test_loss = []
train_loss = []

with open(log, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_accuracies.append(round(float(row['test_acc']), 2))
        train_accuracies.append(round(float(row['train_acc']), 2))
        test_loss.append(round(float(row['test_loss']), 2))
        train_loss.append(round(float(row['train_loss']), 2))


plt.figure('spec_acc_graph')
plt.plot(test_accuracies, color='r', label='test')
plt.plot(train_accuracies, color='g', label='train')
plt.xlabel("Epoca")
plt.ylabel("Acuratețe")
plt.legend()
plt.grid()

plt.figure('spec_loss_graph')
plt.plot(test_loss, color='r', label='test')
plt.plot(train_loss, color='g', label='train')
plt.xlabel("Epoca")
plt.ylabel("Cost")
plt.legend()
plt.grid()

plt.show()


# import matplotlib.pyplot as plt
# import torchaudio
# import numpy as np
# from matplotlib.ticker import FormatStrFormatter
#
# from util.ioUtil import get_spectrogram_from_waveform_in_db
#
# file1 = "/home/dragos/Desktop/Facultate/Licenta/CREMA-D/AudioMP3/1010_DFA_DIS_XX.mp3"
# file2 = "/home/dragos/Desktop/Facultate/Licenta/CREMA-D/AudioMP3/1009_DFA_SAD_XX.mp3"
#
# waveform_data1, sample_rate1 = torchaudio.load(file1)
# waveform_data2, sample_rate2 = torchaudio.load(file2)
#
# waveform1 = waveform_data1[0]
# waveform2 = waveform_data2[0]
#
# plt.figure('1001_TIE_HAP_XX')
# plt.xlabel('Timp [s]')
# plt.ylabel('Amplitudine')
# xpoints1 = range(len(waveform1))
# xshownpoints1 = np.divide(xpoints1, sample_rate1)
# plt.plot(xshownpoints1, waveform1)
#
# plt.figure('1010_DFA_DIS_XX')
# plt.xlabel('Timp [s]')
# plt.ylabel('Amplitudine')
# xpoints2 = range(len(waveform2))
# xshownpoints2 = np.divide(xpoints2, sample_rate2)
# plt.plot(xshownpoints2, waveform2)
#
# spec_fig = plt.figure('spec_1010_DFA_DIS_XX')
# plt.xlabel('Timp [s]')
# plt.ylabel('Frecvența [Hz]')
# start_slice = 0.05
# end_slice = 0.95
# no_frames = len(waveform1)
# start = int(no_frames * start_slice)
# end = int(no_frames * end_slice)
# waveform_slice = waveform1[start:end]
# spec1 = get_spectrogram_from_waveform_in_db(waveform_slice)
#
# n_fft = 512
# hop_length = 128
#
# frame_duration = hop_length / sample_rate1
# xshownpoints1 = np.multiply(range(spec1.shape[1]), frame_duration)
# xshownpoints1 = np.round(xshownpoints1, 2)
#
# spec1 = spec1[:150, :]
#
# spec_img = plt.imshow(spec1)
# plt.xticks(range(spec1.shape[1]), xshownpoints1)
# ax = plt.gca()
# ax.invert_yaxis()
# custom_ticks = ax.get_xticks()[::100]
# ax.set_xticks(custom_ticks)
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# clbr = plt.colorbar(spec_img, shrink=0.4)
# clbr.ax.set_title('Putere [dB]')
# plt.xticks(ax.get_xticks(), xshownpoints1[::100])
# plt.tight_layout()
# plt.show()

