import os
import cv2
import random
import numpy as np

class FrameSampler:
    def __init__(self, train_videos, test_videos, output_dir, num_frames=10, frame_size=(224, 224)):
        self.train_videos = train_videos
        self.test_videos = test_videos
        self.output_dir = output_dir
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.dirs = {
            "train_label_0": os.path.join(self.output_dir, "train_label_0"),
            "train_label_1": os.path.join(self.output_dir, "train_label_1"),
            "test_label_0": os.path.join(self.output_dir, "test_label_0"),
            "test_label_1": os.path.join(self.output_dir, "test_label_1"),
        }

    def sample_frames(self):
        if not self._check_dirs_exist():
            self._create_dirs()
            self._sample_and_save_frames(self.train_videos, "train")
            self._sample_and_save_frames(self.test_videos, "test")

    def _check_dirs_exist(self):
        return all(os.path.exists(dir_path) for dir_path in self.dirs.values())

    def _create_dirs(self):
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)

    def _sample_and_save_frames(self, video_list, split):
        for video_path, label in video_list:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = random.sample(range(total_frames), min(self.num_frames, total_frames))
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.frame_size)
                    frame_filename = f"{split}_frame_{os.path.basename(video_path)}_{idx}.jpg"
                    label_dir = self.dirs[f"{split}_label_{label}"]
                    frame_path = os.path.join(label_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
            cap.release()
