import torch

from dataloader import (
    Compressor, 
    VideoDatasetSplitter, 
    FrameSampler, 
    VideoFrameDataset,
    FrameLoader
)

from classification import (
    Classificator,
    Trainer,
    Evaluator
)

from config import (
    dataset_config, 
    training_config
)

import matplotlib.pyplot as plt
import numpy as np

class Training:

    def __init__(self):
        
        compressor = Compressor(
            dataset_config.vid_original_dir, 
            dataset_config.vid_compressed_dir
        )
        compressor.compress_videos()
        
        video_dataset_splitter = VideoDatasetSplitter(
            original_dir=dataset_config.vid_original_dir, 
            compressed_dir=dataset_config.vid_compressed_dir,
            test_size=dataset_config.test_size,
            random_state=42
        )
        train_videos, test_videos = video_dataset_splitter.split_dataset()
        
        sampler = FrameSampler(
            train_videos=train_videos, 
            test_videos=test_videos,
            output_dir=dataset_config.frames_dir,
            frame_size=dataset_config.frame_size
        )
        sampler.sample_frames()

        loader = FrameLoader(input_dir=dataset_config.frames_dir)
        (
            train_frames, 
            train_labels, 
            test_frames, 
            test_labels
        ) = loader.load_frames()

        train_dataset = VideoFrameDataset(train_frames, train_labels)
        test_dataset = VideoFrameDataset(test_frames, test_labels)

        model = Classificator(num_classes=training_config.num_classes)  
        criterion = torch.nn.BCEWithLogitsLoss()  
        trainer = Trainer(
            model=model, 
            criterion=criterion, 
            batch_size=training_config.batch_size
        )

        trained_model, loss_history = trainer.train(
            train_dataset=train_dataset, 
            num_epochs=training_config.number_of_epochs
        )

        evaluator = Evaluator(
            model=model, 
            batch_size=training_config.batch_size
        )
        accuracy, auc, precision, f1 = evaluator.evaluate(test_dataset=test_dataset)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # plt.plot(loss_history)
        # plt.show()

        torch.save(
            trained_model.state_dict(), 
            training_config.save_model_path
        )
        print(f'Model saved to {training_config.save_model_path}')


