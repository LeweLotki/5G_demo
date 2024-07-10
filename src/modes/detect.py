import torch
from detection import MTNetwork


class Detect:

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mt_network = MTNetwork()        

    def detect(self, image_path, threshold):
        
        probability = self.mt_network.detect(image_path)

        is_detectable = probability > threshold

        print(f'probability of detected face: {probability}')
        print(f'is face detectable(with given threshold): {is_detectable}')

