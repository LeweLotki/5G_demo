import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights

class Classificator(nn.Module):
    def __init__(self, num_classes=2):  # Set num_classes to 1 for binary classification
        super(Classificator, self).__init__()
        
        # Load the pre-trained ResNet18 model with the correct weights parameter
        self.base_model = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Freeze all the convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Replace the fully connected layer with custom layers
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes - 1)  # Output is single value for binary classification
        )
        
        # Set the optimizer to only update the new fully connected layers
        self.optimizer = optim.Adam(self.base_model.fc.parameters(), lr=0.1)

    def forward(self, x):
        x = self.base_model(x)
        return x
