
import torch
from dataloader import TransformedData, Compressor
from classification import Classificator, Visualizer, Trainer
from config import dataset_config, training_config

def main():
    
    compressor = Compressor(
        dataset_config.vid_original_dir, 
        dataset_config.vid_compressed_dir
    )
    compressor.compress_videos()
    
    # data = TransformedData(
    #     img_dir=dataset_config.img_dir, 
    #     max_size=dataset_config.number_of_images
    # )

    
    # model = Classificator(num_classes=1)  
    # criterion = torch.nn.BCEWithLogitsLoss()  
    # visualizer = Visualizer()
    # trainer = Trainer(
    #     model=model, 
    #     criterion=criterion, 
    #     visualizer=visualizer, 
    #     batch_size=training_config.batch_size
    # )

    
    # trained_model = trainer.train(
    #     dataset=data.get_dataset(), 
    #     num_epochs=training_config.number_of_epochs
    # )

    
    # torch.save(
    #     trained_model.state_dict(), 
    #     training_config.save_model_path
    # )
    # print(f'Model saved to {training_config.save_model_path}')

if __name__ == '__main__':
    main()
