import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class Trainer:
    def __init__(self, model, criterion, batch_size, learning_rate=0.001):
        self.model = model
        self.criterion = criterion
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def train(self, train_dataset, num_epochs):
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 99:    # Print every 100 mini-batches
                    print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
                    
            print(f'Epoch [{epoch + 1}/{num_epochs}] completed. Loss: {running_loss:.4f}')
        
        return self.model

