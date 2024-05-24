import os
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from .data import Data  # Adjust the import according to your file structure
import torchvision.transforms.functional as F

class TransformedData(Data):
    def __init__(self, img_dir, max_size=5000):
        super().__init__(img_dir, max_size)
        self.default_transform = self.get_default_transform()
        self.blurred_transform = self.get_blurred_transform()

    def __len__(self):
        return len(self.img_names) * 2

    def __getitem__(self, idx):
        original_len = len(self.img_names)
        if idx < original_len:
            image = super().__getitem__(idx)
            image = F.to_pil_image(image)
            image = self.default_transform(image)
        else:
            image = super().__getitem__(idx - original_len)
            image = F.to_pil_image(image)
            image = self.blurred_transform(image)

        return image

    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
        ])
    
    @staticmethod
    def get_blurred_transform():
        return transforms.Compose([
            transforms.Resize((720, 720)),
            transforms.Lambda(lambda img: TransformedData.blur_background(img)),
            transforms.ToTensor(),
        ])
    
    @staticmethod
    def blur_background(image):
        # Convert PIL image to numpy array
        img_np = np.array(image)
        # Ensure the image is in uint8 format
        if img_np.dtype != np.uint8:
            img_np = img_np.astype(np.uint8)
        # Convert RGB to BGR for OpenCV
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        # Apply Gaussian blur to the whole image
        blurred_img_np = cv2.GaussianBlur(img_np, (21, 21), 0)
        # Convert back to RGB
        blurred_img_np = cv2.cvtColor(blurred_img_np, cv2.COLOR_BGR2RGB)
        # Convert numpy array back to PIL image
        blurred_image = Image.fromarray(blurred_img_np)
        return blurred_image
