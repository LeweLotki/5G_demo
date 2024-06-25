
import torch
from dataloader import TransformedData, Compressor, VideoDatasetSplitter
from classification import Classificator, Visualizer, Trainer
from config import dataset_config, training_config

def main():
    
    compressor = Compressor(
        dataset_config.vid_original_dir, 
        dataset_config.vid_compressed_dir
    )
    compressor.compress_videos()
    
    video_dataset_splitter = VideoDatasetSplitter(
        original_dir=dataset_config.vid_original_dir, 
        compressed_dir=dataset_config.vid_compressed_dir,
        test_size=0.2, 
        random_state=42
    )
    train_videos, test_videos = video_dataset_splitter.split_dataset()
    
    
    
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
