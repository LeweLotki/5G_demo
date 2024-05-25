import os
from PIL import Image
import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

class TransformedData:
    
    def __init__(self, img_dir, max_size=5000):
        
        images, labels = self.__load_images_and_labels(
            img_dir=img_dir, 
            max_size=max_size
        )
        
        images = self.__transform_images(images)
        
        self.images = torch.stack(images)
        self.labels = torch.tensor(labels)
        
        self.dataset = TensorDataset(self.images, self.labels)
        self.dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
    
    def get_dataloader(self):
        return self.dataloader
    
    def get_dataset(self):
        return self.dataset
    
    def __load_images_and_labels(self, img_dir, max_size=5000):
        img_names = os.listdir(img_dir)[:max_size]
        labels = []
        images = []

        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(img_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            # Randomly decide on the type and severity of distortion
            if idx < max_size // 2:
                image, label = self.__apply_mild_or_background_distortion(image)
            else:
                image, label = self.__apply_severe_distortion(image)
            
            images.append(image)
            labels.append(label)
        
        return images, labels

    def __apply_mild_or_background_distortion(self, image):
        # Randomly choose between mild or background distortion
        if random.random() < 0.5:
            return self.__apply_mild_distortion(image), 0
        else:
            return self.__apply_background_distortion(image), 0

    def __apply_mild_distortion(self, image):
        img_np = np.array(image)
        noise_level = random.randint(5, 15)  # Mild noise level
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        img_np[..., 1] = img_np[..., 1] // random.uniform(1.2, 1.5)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(10, 70)]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, 1)
        return Image.fromarray(img_np)

    def __apply_background_distortion(self, image):
        img_np = np.array(image)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        _, mask = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        bg = cv2.GaussianBlur(img_np, (25, 25), 50)
        fg = cv2.bitwise_and(img_np, img_np, mask=mask)
        combined = cv2.add(bg, fg)
        return Image.fromarray(combined)

    def __apply_severe_distortion(self, image):
        img_np = np.array(image)
        noise_level = random.randint(0, 1)  # Higher noise level for severe distortion
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(80, 100)]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, 1)
        return Image.fromarray(img_np), 1
    5
    def __transform_images(self, images):
        transform = transforms.Compose([
            transforms.Resize((720, 720)),
            transforms.ToTensor(),
        ])
        return [transform(image) for image in images]
