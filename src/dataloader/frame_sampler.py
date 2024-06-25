import os
import cv2
import random
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class FrameSampler:
    def __init__(self, train_videos, test_videos, num_frames=10, frame_size=(224, 224)):
        self.train_videos = train_videos
        self.test_videos = test_videos
        self.num_frames = num_frames
        self.frame_size = frame_size

    def sample_frames(self):
        train_frames, train_labels = self._sample_frames_from_videos(self.train_videos)
        test_frames, test_labels = self._sample_frames_from_videos(self.test_videos)
        return train_frames, train_labels, test_frames, test_labels

    def _sample_frames_from_videos(self, video_list):
        frames = []
        labels = []
        for video_path, label in video_list:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = random.sample(range(total_frames), min(self.num_frames, total_frames))
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.resize(frame, self.frame_size)  # Resize frame to the specified size
                    frames.append(frame)
                    labels.append(label)
            cap.release()
        return frames, labels

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)  # Make sure label has the same shape as model output
        return frame, label
