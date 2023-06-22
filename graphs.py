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

# file1 = 'ImageData/1013/1013_DFA_DIS_XX_frame_42.npy'
# file2 = 'ImageData/1044/1044_IEO_HAP_MD_frame_42.npy'
#
# frame1 = np.load(file1)
# frame2 = np.load(file2)
#
# plt.figure('1013_DFA_DIS_XX'), plt.imshow(frame1[:, :, ::-1])
# plt.figure('1044_IEO_HAP_MD'), plt.imshow(frame2[:, :, ::-1])
# plt.show()
