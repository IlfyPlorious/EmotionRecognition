class VideoFile:
    def __init__(self, sample, actor, emotion, emotion_level, video_data):
        self.sample = sample
        self.actor = actor
        self.emotion = emotion
        self.emotion_level = emotion_level
        self.video_data = video_data

    def get_number_of_frames(self):
        duration = self.metadata['video']['duration']
        fps = self.metadata['video']['fps']
        return duration * fps

    def get_file_name(self):
        return f'{self.actor}_{self.sample}_{self.emotion}_{self.emotion_level}'

    def __str__(self) -> str:
        return f'Sample: {self.sample}, actor: {self.actor}, emotion: {self.emotion},\n'
