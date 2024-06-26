import os
import cv2
import torch
from torchvision import transforms

class FrameLoader:
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.dirs = {
            "train_label_0": os.path.join(self.input_dir, "train_label_0"),
            "train_label_1": os.path.join(self.input_dir, "train_label_1"),
            "test_label_0": os.path.join(self.input_dir, "test_label_0"),
            "test_label_1": os.path.join(self.input_dir, "test_label_1"),
        }

    def load_frames(self):
        train_frames, train_labels = self._load_frames_from_dirs("train")
        test_frames, test_labels = self._load_frames_from_dirs("test")
        return train_frames, train_labels, test_frames, test_labels

    def _load_frames_from_dirs(self, split):
        frames = []
        labels = []
        for label in [0, 1]:
            dir_path = self.dirs[f"{split}_label_{label}"]
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                frame = cv2.imread(file_path)
                if frame is None:
                    print(f"Warning: Failed to load image {file_path}")
                    continue
                try:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    frame = transforms.ToTensor()(frame)  # Convert to tensor and normalize to [0, 1]
                    frames.append(frame)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")
                    continue
        return frames, labels
