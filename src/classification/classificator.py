import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

class Classificator(nn.Module):
    def __init__(self, num_classes=1):  # Set num_classes to 1 for binary classification
        super(Classificator, self).__init__()

        # Load the InceptionResnetV1 model pre-trained on VGGFace2
        self.base_model = InceptionResnetV1(pretrained='vggface2', classify=True)

        # Get the number of features output by the last layer before the classifier
        num_ftrs = self.base_model.logits.in_features

        # Replace the final layer with custom layers for binary classification
        self.base_model.logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Output is single value for binary classification
        )

        # Set the optimizer to update all parameters, including the convolutional layers
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.base_model(x)
        return x

# Example usage:
# model = Classificator(num_classes=1)
