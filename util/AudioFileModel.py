class AudioFile:
    def __init__(self, sample, actor, emotion, emotion_level, metadata, waveform_data, sample_rate):
        self.sample = sample
        self.actor = actor
        self.emotion = emotion
        self.emotion_level = emotion_level
        self.metadata = metadata
        self.waveform_data = waveform_data
        self.sample_rate = sample_rate

    def get_length_in_seconds(self):
        return self.waveform_data.shape[1] / self.sample_rate

    def get_number_of_channels(self):
        try:
            return self.waveform_data.shape[0]
        except BaseException:
            return 1

    def get_file_name(self):
        return f'{self.actor}_{self.sample}_{self.emotion}_{self.emotion_level}'

    def __str__(self) -> str:
        return f'Sample: {self.sample}, actor: {self.actor}, emotion: {self.emotion},\n' \
               f' metadata:{self.metadata}, length:{self.get_length_in_seconds()}'
