import json
import os
from os.path import join

import cv2
import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from IPython.display import Audio, display
from PIL import Image
from facenet_pytorch import MTCNN

from networks_files.hook import Hook
from util import AudioFileModel
from util.AudioFileModel import AudioFile

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
crema_d_dir = os.path.join(parent_dir, 'CREMA-D')
audio_wav_dir = os.path.join(crema_d_dir, 'AudioWAV')

labels = {
    'ANGER': 0,
    'DISGUST': 1,
    'FEAR': 2,
    'HAPPY': 3,
    'NEUTRAL': 4,
    'SAD': 5
}

get_labels = {
    0: 'ANGER',
    1: 'DISGUST',
    2: 'FEAR',
    3: 'HAPPY',
    4: 'NEUTRAL',
    5: 'SAD'
}
config = json.load(open('config.json'))


def get_audio_files(limit=4000, more_than_3_seconds=False) -> list[AudioFileModel.AudioFile]:
    wav_files = os.listdir(audio_wav_dir)
    audio_files_list_less_3 = list()
    audio_files_list_more_3 = list()
    for file in wav_files[:limit]:
        actor, sample, emotion, emotion_level = file.split('_')
        emotion = get_emotion_by_notation(emotion)
        emotion_level = get_emotion_level_by_notation(emotion_level)
        wav_file_path = join(audio_wav_dir, file)
        metadata = torchaudio.info(wav_file_path)
        waveform_data, sample_rate = torchaudio.load(wav_file_path)
        audio_file = AudioFileModel.AudioFile(sample=sample, actor=actor, emotion=emotion,
                                              emotion_level=emotion_level, metadata=metadata,
                                              waveform_data=waveform_data, sample_rate=sample_rate)
        if audio_file.get_length_in_seconds() < 3:
            audio_files_list_less_3.append(audio_file)
        else:
            audio_files_list_more_3.append(audio_file)

    return audio_files_list_more_3 if more_than_3_seconds else audio_files_list_less_3


def get_spectrogram_from_waveform_in_db(waveform: np.ndarray, stretched=False) -> np.ndarray:
    n_fft = 512
    win_length = None
    hop_length = 128

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    # Perform transformation
    if stretched:
        final_spectrogram = stretch_spectrogram(spectrogram(waveform), n_freq=n_fft // 2 + 1, hop_length=hop_length)
    else:
        final_spectrogram = spectrogram(waveform)

    return librosa.power_to_db(final_spectrogram)


def stretch_spectrogram(spectrogram, index=1, final_dim=200, n_freq=201, hop_length=None):
    stretch = T.TimeStretch(n_freq=n_freq, hop_length=hop_length)
    rate = len(spectrogram[0][index]) / final_dim
    stretched_spectrogram = stretch(spectrogram, rate)
    if stretched_spectrogram.shape[2] != 100:
        rate = len(stretched_spectrogram[0][index]) / final_dim
        stretched_spectrogram = stretch(stretched_spectrogram, rate)

    return stretched_spectrogram


def save_spectrogram_data(file: AudioFile, save_dir='SpectrogramData', stretched=False, start_slice=0.05,
                          end_slice=0.95):
    no_frames = len(file.waveform_data[0])
    start = int(no_frames * start_slice)
    end = int(no_frames * end_slice)
    waveform_slice = file.waveform_data[:, start:end]
    spectrogram = get_spectrogram_from_waveform_in_db(waveform_slice, stretched=stretched)
    actor_dir = os.path.join(save_dir, file.actor)
    saved_file_name = os.path.join(actor_dir, file.get_file_name())

    if not os.path.exists(actor_dir):
        os.makedirs(actor_dir)

    np.save(saved_file_name, spectrogram)


def get_emotion_level_by_notation(notation: str) -> str:
    if notation == 'LO':
        return "LOW"
    elif notation == 'MD':
        return "MEDIUM"
    elif notation == 'HI':
        return "HIGH"
    else:
        return "UNSPECIFIED"


def get_notation_by_emotion_level(emotion_level: str) -> str:
    if emotion_level == 'LOW':
        return "LO"
    elif emotion_level == 'MEDIUM':
        return "MD"
    elif emotion_level == 'HIGH':
        return "HI"
    else:
        return "XX"


def get_emotion_by_notation(notation: str) -> str:
    if notation == 'ANG':
        return "ANGER"
    elif notation == 'DIS':
        return "DISGUST"
    elif notation == 'FEA':
        return "FEAR"
    elif notation == 'HAP':
        return "HAPPY"
    elif notation == 'SAD':
        return "SAD"
    else:
        return "NEUTRAL"


def get_notation_by_emotion(emotion: str) -> str:
    if emotion == 'ANGER':
        return "ANG"
    elif emotion == 'DISGUST':
        return "DIS"
    elif emotion == 'FEAR':
        return "FEA"
    elif emotion == 'HAPPY':
        return "HAP"
    elif emotion == 'SAD':
        return "SAD"
    else:
        return "NEU"


def map_tensor_to_0_1(tensor: torch.Tensor) -> torch.Tensor:
    minimum = torch.min(tensor)
    maximum = torch.max(tensor)
    return torch.div(tensor - minimum, maximum - minimum)


def map_to_0_1(matrix):
    minimum = np.min(matrix)
    maximum = np.max(matrix)
    return (matrix - minimum) / (maximum - minimum)


def save_image_data(data, parent_dir, dir, file_name):
    print(f'Saving file {file_name}...')
    if parent_dir is None:
        pass
    else:
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)

        actor_dir = os.path.join(parent_dir, dir)

        if not os.path.exists(actor_dir):
            os.makedirs(actor_dir)

        np.save(os.path.join(actor_dir, file_name), np.array(data))


def write_video_frames_as_npy():
    video_dir_path = config['video_dir_path']
    videos = os.listdir(video_dir_path)
    for actor in range(1001, 1092):
        videos_for_actor = list(filter(lambda video_name: str(actor) in video_name, videos))
        print(f'-------- Saving for actor: {actor} ---------')
        for file in videos_for_actor:
            video_file = os.path.join(video_dir_path, file)
            video_capture = cv2.VideoCapture(video_file)
            total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

            step = int(total_frames // 10)

            start = int(total_frames * 0.07)
            end = int(total_frames * 0.9)

            i = start
            while i < end:
                ret, frame = video_capture.read()

                try:
                    if i % step == 0:
                        dir_file = file.split("_")[0]
                        file_name = file.split(".")[0] + f'_frame_{i}'
                        try:
                            crop = get_face_cropped_image(frame)
                            resize = cv2.resize(crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                            save_image_data(data=map_to_0_1(resize), parent_dir='ImageData', dir=dir_file,
                                            file_name=file_name)
                        except:
                            print(f'Failed saving file {file}')
                except:
                    print('Something went wrong')

                i += 1

            video_capture.release()


def get_face_cropped_image(img):
    pil_img = Image.fromarray(img.astype(np.uint8))

    mtcnn = MTCNN()

    # # Get cropped and prewhitened image tensor
    coord, _ = mtcnn.detect(pil_img)
    y_tl = int(coord[0][0])  # top left y coordinate
    x_tl = int(coord[0][1])  # top left x coordinate
    y_br = int(coord[0][2])  # bottom right y coordinate
    x_br = int(coord[0][3])  # bottom right x coordinate

    return img[x_tl:x_br, y_tl:y_br, :]


def spectrogram_windowing(spectrogram, window_size=70, window_count=5, plot=False):
    channels, frequencies, timespan = spectrogram.shape
    windows = torch.empty((window_count, channels, frequencies, window_size))
    windows_indexes = list()

    if window_count == 1:
        return [spectrogram[:, :, :window_size]]

    window_overlap = int((window_count * window_size - timespan) / (window_count - 1))

    for i in range(0, window_count - 1):
        windows[i] = spectrogram[:, :,
                     i * (window_size - window_overlap): i * (window_size - window_overlap) + window_size]

        windows_indexes.append((i * (window_size - window_overlap), i * (window_size - window_overlap) + window_size))

    windows[window_count - 1] = spectrogram[:, :, timespan - window_size:]
    windows_indexes.append((timespan - window_size, spectrogram.shape[2]))

    if plot:
        plt.figure(), plt.subplot(window_count + 1, 1, 1), plt.imshow(torch.permute(spectrogram, (1, 2, 0)))
        for i in range(0, window_count):
            plt.subplot(window_count + 1, 1, i + 2), plt.imshow(torch.permute(windows[i], (1, 2, 0)))

        plt.show()

    return windows, windows_indexes


def random_spectrogram_windowing(spectrogram, window_size=70, window_count=5, plot=False):
    channels, frequencies, timespan = spectrogram.shape
    windows = torch.empty((window_count, channels, frequencies, window_size))

    for i in range(0, window_count):
        start = np.random.randint(low=0, high=timespan - window_size)
        end = start + window_size
        windows[i] = spectrogram[:, :, start:end]

    if plot:
        plt.figure(), plt.subplot(window_count + 1, 1, 1), plt.imshow(torch.permute(spectrogram, (1, 2, 0)))
        for i in range(0, window_count):
            plt.subplot(window_count + 1, 1, i + 2), plt.imshow(torch.permute(windows[i], (1, 2, 0)))

        plt.show()

    return windows


def try_to_get_valid_video_capture(path, intensity='XX'):
    video_capture = cv2.VideoCapture(path)
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    if total_frames == 0:
        video_capture.release()
        video_path_split = path.split('_')
        video_path_split[-1] = f'{intensity}.flv'

        new_path = '_'.join(video_path_split)

        video_capture = cv2.VideoCapture(new_path)
        total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

    return video_capture, total_frames


def compute_entropy(probabilities):
    entropies = np.zeros(len(probabilities))
    for idx in range(len(probabilities)):
        for probability in probabilities[idx]:
            entropies[idx] = entropies[idx] - probability * np.log2(probability)
    return entropies


def spectrogram_name_splitter(spec_name):
    actor, line, emotion, intensity = spec_name.split('_')
    intensity = intensity.split('.')[0]
    extension = intensity.split('.')[-1]

    emotion = get_notation_by_emotion(emotion)
    intensity = get_notation_by_emotion_level(intensity)

    return actor, line, emotion, intensity, extension


def write_pretrained_model_features_for_video(model, start=1001, end=1092):
    video_dir_path = config['video_dir_path']
    videos = os.listdir(video_dir_path)
    for actor in range(start, end):
        videos_for_actor = list(filter(lambda video_name: str(actor) in video_name, videos))
        print(f'-------- Saving for actor: {actor} ---------')

        for file in videos_for_actor:
            video_file = os.path.join(video_dir_path, file)
            video_capture = cv2.VideoCapture(video_file)
            total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)

            start = int(total_frames * 0.07)
            end = int(total_frames * 0.9)

            model = model.to('cuda')

            i = start
            while i < end:
                ret, frame = video_capture.read()

                dir_file = file.split("_")[0]
                file_name = file.split(".")[0] + f'_frame_{i}'
                try:
                    crop = get_face_cropped_image(frame)
                    resize = cv2.resize(crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                    resize = map_to_0_1(resize)
                    features = None
                    hook = Hook()

                    layer = model.get_submodule('avgpool')
                    handle = layer.register_forward_hook(hook)

                    frame_input = torch.tensor([np.transpose(resize, (2, 0, 1))]).float().cuda()
                    with torch.no_grad():
                        _ = model(frame_input.cuda())

                    features = hook.outputs[0].squeeze()
                    hook.clear()
                    save_image_data(data=features.cpu().detach().numpy(), parent_dir='FeaturesData', dir=dir_file,
                                    file_name=file_name)
                except Exception as e:
                    print(e)
                    print(f'Failed saving file {file}')

                i += 1

            video_capture.release()
