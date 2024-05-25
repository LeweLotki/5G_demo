import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import TransformedData
from classification import Classificator, Visualizer


def main():
    # Load data
    img_dir = '../data/Humans'
    data = TransformedData(img_dir, max_size=20)
    dataloader = data.get_dataloader()

    # Initialize the model, loss function, and visualizer
    model = Classificator(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    visualizer = Visualizer()

    # Training the model
    num_epochs = 5
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader, 0):
            # Zero the parameter gradients
            model.optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Backward pass and optimization
            loss.backward()
            model.optimizer.step()

            # Update metrics
            visualizer.update_metrics(outputs, labels, loss.item())

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # Plot metrics
    visualizer.plot_metrics()

    # Calculate final metrics on the whole dataset
    visualizer.calculate_final_metrics(model, dataloader)

    # Display some images and their predicted and actual labels
    visualizer.show_images_with_labels(dataloader, model)

if __name__ == '__main__':
    main()
