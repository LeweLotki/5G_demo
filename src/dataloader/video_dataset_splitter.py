import os
from sklearn.model_selection import train_test_split

class VideoDatasetSplitter:
    def __init__(self, original_dir, compressed_dir, test_size=0.2, random_state=42):
        self.original_dir = original_dir
        self.compressed_dir = compressed_dir
        self.test_size = test_size
        self.random_state = random_state
        self.supported_formats = (".mp4", ".avi", ".mov")  # Add other video formats as needed

    def get_video_paths(self, directory):
        video_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith(self.supported_formats):
                    video_paths.append(os.path.join(root, file))
        return video_paths

    def split_dataset(self):
        original_videos = self.get_video_paths(self.original_dir)
        compressed_videos = self.get_video_paths(self.compressed_dir)
        
        original_train, original_test = train_test_split(original_videos, test_size=self.test_size, random_state=self.random_state)
        compressed_train, compressed_test = train_test_split(compressed_videos, test_size=self.test_size, random_state=self.random_state)
        
        train_videos = [(video, 0) for video in original_train] + [(video, 1) for video in compressed_train]
        test_videos = [(video, 0) for video in original_test] + [(video, 1) for video in compressed_test]

        return train_videos, test_videos
    
