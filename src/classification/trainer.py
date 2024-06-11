import torch
from torch.utils.data import DataLoader, random_split

class Trainer:
    def __init__(self, model, criterion, visualizer, batch_size=32, test_size=0.2):
        self.model = model
        self.criterion = criterion
        self.visualizer = visualizer
        self.batch_size = batch_size
        self.test_size = test_size

        # Set the device to CPU
        self.device = torch.device("cpu")
        self.model.to(self.device)  # Move the model to the appropriate device

    def train(self, dataset, num_epochs):
        # Split dataset into training and testing sets
        train_size = int((1 - self.test_size) * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        # Training the model
        for epoch in range(num_epochs):
            running_loss = 0.0
            self.model.train()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)  # Ensure labels are of shape (batch_size, 1)

                # Zero the parameter gradients
                self.model.optimizer.zero_grad()

                # Forward pass
                outputs = self.model(inputs)

                loss = self.criterion(outputs, labels)
                # Backward pass and optimization
                loss.backward()
                self.model.optimizer.step()

                # Update metrics
                self.visualizer.update_metrics(outputs, labels, loss.item())

                # Print statistics
                running_loss += loss.item()
                if (i + 1) % self.batch_size == 0:  # Print every batch_size mini-batches
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / self.batch_size:.3f}')
                    running_loss = 0.0

            # Evaluate metrics at the end of each epoch
            self.evaluate_epoch(epoch, train_loader, test_loader)

        print('Finished Training')

        # Plot metrics
        self.visualizer.plot_metrics()

        # Calculate final metrics on the testing dataset
        self.visualizer.calculate_final_metrics(self.model, test_loader)

        # Display some images and their predicted and actual labels from the test set
        self.visualizer.show_images_with_labels(test_loader, self.model)

        return self.model

    def evaluate_epoch(self, epoch, train_loader, test_loader):
        self.model.eval()
        with torch.no_grad():
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)  # Ensure labels are of shape (batch_size, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                train_loss += loss.item()
                preds = (outputs > 0.5).float()
                train_correct += (preds == labels).sum().item()
            train_loss /= len(train_loader)
            train_accuracy = train_correct / len(train_loader.dataset)

            test_loss, test_correct = 0.0, 0
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device).float().view(-1, 1)  # Ensure labels are of shape (batch_size, 1)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                preds = (outputs > 0.5).float()
                test_correct += (preds == labels).sum().item()
            test_loss /= len(test_loader)
            test_accuracy = test_correct / len(test_loader.dataset)

            print(f'Epoch {epoch + 1}: Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}')
