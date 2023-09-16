import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
from torch import cuda

from networks_files.brain import Brain
from networks_files.hook import Hook
from networks_files.networks import SpectrogramBrain
from networks_files.res_net import BasicBlock
from util.ioUtil import get_spectrogram_from_waveform_in_db, map_to_0_1, \
    compute_entropy, spectrogram_windowing, get_face_cropped_image, initialize_model, save_image_data, get_labels

device = "cuda" if cuda.is_available() else "cpu"
config = json.load(open('config.json'))

audio_file = "/home/dragos/Desktop/Facultate/Licenta/demo/demo.mp3"
video_file = "/home/dragos/Desktop/Facultate/Licenta/demo/demo.mp4"
frames_dir = "/home/dragos/Desktop/Facultate/Licenta/demo/frames"

waveform_data, sample_rate = torchaudio.load(audio_file)
waveform = waveform_data[0]
spec = get_spectrogram_from_waveform_in_db(waveform)
spec = map_to_0_1(spec)

plt.figure('spectrogram')
plt.imshow(spec)

# init models

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
brain_model.load_state_dict(brain_checkpoint['model_weights'])

spec_model = SpectrogramBrain(block=BasicBlock, layers=[2, 2, 3, 2], num_classes=6).to(device)
spec_model.load_state_dict(spec_checkpoint['model_weights'])

windows, windows_indexes = spectrogram_windowing(torch.tensor([spec]), window_size=120,
                                                 window_count=3)
with torch.no_grad():
    spec_prediction, spec_features_layer = spec_model(windows.to(device))

print(spec_prediction)
print('Predictii per fereastra')

spec_prediction = spec_prediction.detach().cpu().numpy()

for i in range(3):
    print(f'Fereastra {i + 1}: {get_labels[np.argmax(spec_prediction[i])]}')

entropies = compute_entropy(spec_prediction)

best_window_index = np.argmin(entropies)

start, end = windows_indexes[best_window_index]
print(f'Fereastra cea mai reprezentativa: {best_window_index + 1}')


def save_video_frames():
    ## --------- SAVE THE VIDEO FRAMES --------- ##
    video_capture = cv2.VideoCapture(video_file)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    start = int(total_frames * 0.07)
    end = int(total_frames * 0.93)

    model = initialize_model(6, True)
    model = model.to(device)

    i = start
    while i < end:
        ret, frame = video_capture.read()
        file_name = f'frame_{i if i > 9 else f"0{i}"}'
        try:
            crop = get_face_cropped_image(frame)
            resize = cv2.resize(crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            resize = map_to_0_1(resize)
            cropped_face = resize[:, :, ::-1]
            features = None
            hook = Hook()

            layer = model.get_submodule('avgpool')
            handle = layer.register_forward_hook(hook)

            frame_input = torch.tensor([np.transpose(resize, (2, 0, 1))]).float().cuda()
            with torch.no_grad():
                _ = model(frame_input.cuda())

            features = hook.outputs[0].squeeze()
            hook.clear()
            save_image_data(data=features.cpu().detach().numpy(), parent_dir='FeaturesData', dir=frames_dir,
                            file_name=file_name)
        except Exception as e:
            print(e)
            print(f'Failed saving')

        i += 1

    video_capture.release()

    plt.figure('Face crop')
    plt.imshow(cropped_face)
    plt.show()

print("Saving video frames")
save_video_frames()


def get_frame_features_entropy_based(spec, start, end):
    ## ---------- MANAGE VIDEO FRAME WITH ENTROPY --------- ##

    frames = os.listdir(frames_dir)

    spectrogram_length = spec.shape[1]
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

    try:
        start_stack = []
        mid_stack = []
        end_stack = []
        for frame_name in start_frame_name:
            frame = torch.tensor(np.load(os.path.join(frames_dir, frame_name)))
            start_stack.append(frame)

        for frame_name in mid_frame_name:
            frame = torch.tensor(np.load(os.path.join(frames_dir, frame_name)))
            mid_stack.append(frame)

        for frame_name in end_frame_name:
            frame = torch.tensor(np.load(os.path.join(frames_dir, frame_name)))
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

    return frame_image_features


frame_image_features = get_frame_features_entropy_based(spec, start, end)
print(f'Dimensionalitatea trasaturilor cadrelor: {frame_image_features.shape}')

frame_image_features = frame_image_features.to(device)
spec_features_layer = spec_features_layer[best_window_index]
spec_features_layer = torch.stack([spec_features_layer]).to(device)
output = brain_model(frame_image_features, spec_features_layer).cuda()
print(output)
pred = torch.argmax(output).item()
print(f'Predictia retelei finale: {get_labels[pred]}')
