import torch
from facenet_pytorch import MTCNN
import cv2
import matplotlib.pyplot as plt
from PIL import Image


class MTNetwork:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)

    def detect(self, image_path: str):
        image = Image.open(image_path)
        _, probs = self.mtcnn.detect(image)

        probability = probs[0]

        return probability

