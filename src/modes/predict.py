import torch
import cv2
from torchvision import transforms
from classification import Classificator

class Predict:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Classificator(num_classes=1)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((160, 160)),
        ])

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
        image = image.unsqueeze(0)  
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)
            prediction = torch.sigmoid(output).item()

        return prediction

