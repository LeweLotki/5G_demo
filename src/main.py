
import torch
from dataloader import (
    Compressor, 
    VideoDatasetSplitter, 
    FrameSampler, 
    VideoFrameDataset
)
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
    
    frame_size = (160, 160)
    sampler = FrameSampler(
        train_videos=train_videos, 
        test_videos=test_videos,
        frame_size=frame_size
    )
    train_frames, train_labels, test_frames, test_labels = sampler.sample_frames()
    
    # Creating PyTorch datasets
    train_dataset = VideoFrameDataset(train_frames, train_labels)
    test_dataset = VideoFrameDataset(test_frames, test_labels)

    model = Classificator(num_classes=1)  
    criterion = torch.nn.BCEWithLogitsLoss()  
    trainer = Trainer(
        model=model, 
        criterion=criterion, 
        batch_size=training_config.batch_size
    )

    trained_model = trainer.train(
        train_dataset=train_dataset, 
        num_epochs=training_config.number_of_epochs
    )

    torch.save(
        trained_model.state_dict(), 
        training_config.save_model_path
    )
    print(f'Model saved to {training_config.save_model_path}')

if __name__ == '__main__':
    main()
