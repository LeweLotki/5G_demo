import torch
from torch.utils.data import Dataset

class VideoFrameDataset(Dataset):
    def __init__(self, frames, labels):
        self.frames = frames
        self.labels = labels

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        label = self.labels[idx]
        return frame, torch.tensor([label], dtype=torch.float32)
