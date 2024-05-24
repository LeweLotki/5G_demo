import os
from PIL import Image
import numpy as np
import cv2
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

def load_images_and_labels(img_dir, max_size=5000):
    img_names = os.listdir(img_dir)[:max_size]
    labels = [0] * (max_size // 2) + [1] * (max_size // 2)
    
    images = []
    for idx, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        # Apply distortions based on label
        if labels[idx] == 0:
            image = apply_mild_distortion(image)
        else:
            image = apply_severe_distortion(image)
        
        images.append(image)
    
    return images, labels

def apply_mild_distortion(image):
    img_np = np.array(image)
    noise = np.random.normal(0, 10, img_np.shape).astype(np.uint8)
    img_np = cv2.add(img_np, noise)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    img_np[..., 1] = img_np[..., 1] // 1.5
    img_np = cv2.cvtColor(img_np, cv2.COLOR_HSV2RGB)
    return Image.fromarray(img_np)

def apply_severe_distortion(image):
    img_np = np.array(image)
    noise = np.random.normal(0, 50, img_np.shape).astype(np.uint8)
    img_np = cv2.add(img_np, noise)
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 10]
    _, encimg = cv2.imencode('.jpg', img_np, encode_param)
    img_np = cv2.imdecode(encimg, 1)
    return Image.fromarray(img_np)

def transform_images(images):
    transform = transforms.Compose([
        transforms.Resize((720, 720)),
        transforms.ToTensor(),
    ])
    return [transform(image) for image in images]
