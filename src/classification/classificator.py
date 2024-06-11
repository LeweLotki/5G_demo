import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1

class Classificator(nn.Module):
    def __init__(self, num_classes=1):  
        super(Classificator, self).__init__()

        
        self.base_model = InceptionResnetV1(pretrained='vggface2', classify=True)

        
        num_ftrs = self.base_model.logits.in_features

        
        self.base_model.logits = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  
        )

        
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = self.base_model(x)
        return x



