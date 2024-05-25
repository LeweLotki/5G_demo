# main.py
import torch
from dataloader import TransformedData
from classification import Classificator, Visualizer, Trainer
from config import dataset_config, training_config

def main():
    # Load data
    data = TransformedData(
        img_dir=dataset_config.img_dir, 
        max_size=dataset_config.number_of_images
    )

    # Initialize the model, loss function, and visualizer
    model = Classificator(num_classes=2)
    criterion = torch.nn.CrossEntropyLoss()
    visualizer = Visualizer()
    trainer = Trainer(
        model=model, 
        criterion=criterion, 
        visualizer=visualizer, 
        batch_size=training_config.batch_size
    )

    # Train the model
    trained_model = trainer.train(
        dataset=data.get_dataset(), 
        num_epochs=training_config.number_of_epochs
    )

if __name__ == '__main__':
    main()
