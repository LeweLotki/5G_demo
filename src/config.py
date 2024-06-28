from ypstruct import struct

dataset_config = struct()

dataset_config.vid_original_dir = '../data/videos/original'
dataset_config.vid_compressed_dir = '../data/videos/compressed'
dataset_config.frames_dir = '../data/frames'
dataset_config.frame_size = (160, 160)
dataset_config.test_size = 0.2

training_config = struct()

training_config.num_classes = 1
training_config.number_of_epochs = 20
training_config.batch_size = 10
training_config.save_model_path = '../models/model.pth'
