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
    model = Classificator(num_classes=1)  # Set num_classes to 1 for binary classification
    criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
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

    # Save the model
    torch.save(
        trained_model.state_dict(), 
        training_config.save_model_path
    )
    print(f'Model saved to {training_config.save_model_path}')

if __name__ == '__main__':
    main()
