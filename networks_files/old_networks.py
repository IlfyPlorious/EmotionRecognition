import torch.nn.functional as F
from torch import nn


class EmotionsNetwork2LinLayers(nn.Module):
    def __init__(self):
        super(EmotionsNetwork2LinLayers, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(369 * 496 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
            nn.ReLU(),
            nn.Linear(100, 6)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class EmotionsNetwork2Conv2Layers(nn.Module):
    def __init__(self):
        super(EmotionsNetwork2Conv2Layers, self).__init__()
        # 4 channels because spectrogram tensors shape is [4, 369, 496] which has 4 channels
        self.conv1 = nn.Conv2d(4, 8, 30)
        self.conv2 = nn.Conv2d(8, 16, 30)
        self.fc1 = nn.Linear(4 * 8 * 16, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        # Max pooling over a (6, 6) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (6, 6))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 6)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class EmotionsNetworkV3(nn.Module):
    def __init__(self):
        super(EmotionsNetworkV3, self).__init__()
        # 4 channels because spectrogram tensors shape is [4, 369, 496] which has 4 channels
        self.conv1 = nn.Conv2d(1, 8, 10)
        self.conv2 = nn.Conv2d(8, 16, 10)
        self.fc1 = nn.Linear(192, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        # Max pooling over a (6, 6) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (6, 6))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 6)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    @staticmethod
    def num_flat_features(x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


    def _get_frame_features_random_based(self, spec_path):
        ## ---------- MANAGE VIDEO FRAME RANDOMLY --------- ##

        actor, line, emotion, intensity, _ = spectrogram_name_splitter(spec_path.split('/')[-1])
        image_data_dir = self.config['video_data']
        image_actor_dir = os.path.join(image_data_dir, actor)
        vid_name = f'{actor}_{line}_{emotion}_{intensity}'

        vids = os.listdir(image_actor_dir)
        frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))

        i = 0
        intensities = ['LO', 'MD', 'HI']

        while len(frames) == 0 and i < 3:
            vid_name = f'{actor}_{line}_{emotion}_{intensities[i]}'
            frames = list(filter(lambda actor_vid: vid_name in actor_vid, vids))
            i += 0

        frame_numbers = list(map(lambda frame: int(frame.split('_')[-1].split('.')[0]), frames))
        max_frame_number = np.amax(frame_numbers)
        min_frame_number = np.amin(frame_numbers)

        sorted_frame_numbers = sorted(frames)

        randomizer_1 = np.random.randint(low=min_frame_number, high=max_frame_number - 4)
        randomizer_2 = np.random.randint(low=min_frame_number, high=max_frame_number - 4)
        randomizer_3 = np.random.randint(low=min_frame_number, high=max_frame_number - 4)

        start_frame_name = None
        end_frame_name = None
        mid_frame_name = None

        try:
            start_frame_name = [
                sorted_frame_numbers[randomizer_1],
                sorted_frame_numbers[randomizer_1 + 1],
                sorted_frame_numbers[randomizer_1 + 2]
            ]
            end_frame_name = [
                sorted_frame_numbers[randomizer_2],
                sorted_frame_numbers[randomizer_2 + 1],
                sorted_frame_numbers[randomizer_2 + 2]
            ]
            mid_frame_name = [
                sorted_frame_numbers[randomizer_3],
                sorted_frame_numbers[randomizer_3 + 1],
                sorted_frame_numbers[randomizer_3 + 2]
            ]
        except Exception as e:
            print(e)
            print(f'Problem with {vid_name}')
            print(f'Length of sorted_frame_numbers: {len(sorted_frame_numbers)}')
            print(f'randomizer1: {randomizer_1}')
            print(f'randomizer2: {randomizer_2}')
            print(f'randomizer3: {randomizer_3}')

        try:
            start_stack = []
            mid_stack = []
            end_stack = []
            for frame_name in start_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
                start_stack.append(frame)

            for frame_name in mid_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
                mid_stack.append(frame)

            for frame_name in end_frame_name:
                frame = torch.tensor(np.load(os.path.join(image_actor_dir, frame_name)))
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
            print(f'Problems with {vid_name}')
            print(f'Frame names for {vid_name}: {start_frame_name}, {mid_frame_name}, {end_frame_name}')
            print(f'Frames for {vid_name}: {frame_numbers}')
            print(f'Start index: {start_frame_name}')
            print(f'Mid index: {mid_frame_name}')
            print(f'End index: {end_frame_name}')
            print(f'Spec path {spec_path}')

        return frame_image_features