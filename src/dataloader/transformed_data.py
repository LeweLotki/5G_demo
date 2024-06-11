import os
from PIL import Image
import numpy as np
import cv2
import random
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
import torchvision.transforms.functional as F

class TransformedData:
    
    def __init__(self, img_dir, max_size=5000):
        self.segmentation_model = self.__load_segmentation_model()
        
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
    
    def __load_segmentation_model(self):
        weights = DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
        model = models.segmentation.deeplabv3_resnet101(weights=weights).eval()
        return model
    
    def __load_images_and_labels(self, img_dir, max_size=5000):
        img_names = os.listdir(img_dir)[:max_size]
        labels = []
        images = []

        for idx, img_name in enumerate(img_names):
            img_path = os.path.join(img_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            
            if idx < max_size // 2:
                image, label = self.__apply_mild_or_background_distortion(image)
            else:
                image, label = self.__apply_severe_distortion(image)
            
            images.append(image)
            labels.append(label)
        
        return images, labels

    def __apply_mild_or_background_distortion(self, image):
        
        if random.random() < 0.5:
            return self.__apply_mild_distortion(image), 0
        else:
            return self.__apply_background_distortion(image), 0

    def __apply_mild_distortion(self, image):
        img_np = np.array(image)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(30, 70)]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, 1)
        return Image.fromarray(img_np)

    def __apply_background_distortion(self, image):
        img_np = np.array(image)
        input_tensor = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            output = self.segmentation_model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()
        mask = output_predictions == 15  

        bg = cv2.GaussianBlur(img_np, (15, 15), 10)  
        fg = cv2.bitwise_and(img_np, img_np, mask=mask.astype(np.uint8) * 255)
        combined = cv2.add(bg, fg)
        return Image.fromarray(combined)

    def __apply_severe_distortion(self, image):
        img_np = np.array(image)
        noise_level = random.randint(20, 50)  
        noise = np.random.normal(0, noise_level, img_np.shape).astype(np.uint8)
        img_np = cv2.add(img_np, noise)
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_np = cv2.GaussianBlur(img_np, (5, 5), 0)  
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(1, 10)]
        _, encimg = cv2.imencode('.jpg', img_np, encode_param)
        img_np = cv2.imdecode(encimg, 1)
        return Image.fromarray(img_np), 1

    def __transform_images(self, images):
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
        ])
        return [transform(image) for image in images]
