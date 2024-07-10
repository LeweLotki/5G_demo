import os
import torch
from detection import MTNetwork
from PIL import Image

class Detect:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mt_network = MTNetwork()

    def detect(self, image_path, threshold):
        probability = self.mt_network.detect(image_path)
        if probability:
            is_detectable = probability > threshold
        else:
            print('No face detected')
            is_detectable = False
        print(f'Image: {image_path}')
        print(f'Probability of detected face: {probability}')
        print(f'Is face detectable (with given threshold): {is_detectable}')

    def detect_in_directory(self, dir_path, threshold):
        counter = 0
        for root, _, files in os.walk(dir_path):
            for i, file in enumerate(files):
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_path = os.path.join(root, file)
                    print(image_path)
                    probability = self.mt_network.detect(image_path)
                    print(probability)
                    if probability:
                        is_detectable = probability > threshold
                    else:
                        print('No face detected')
                        is_detectable = False
                    if is_detectable: 
                        counter += 1
                    print(f'Image: {image_path}')
                    print(f'Probability of detected face: {probability}')
                    print(f'Is face detectable (with given threshold): {is_detectable}')
        print(f'percantage of detectable images: {counter / (i + 1)}')

