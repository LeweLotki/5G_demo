from ypstruct import struct

dataset_config = struct()

dataset_config.number_of_images = 100
dataset_config.img_dir = '../data/Humans'

dataset_config.vid_original_dir = '..\\data\\videos\\original'
dataset_config.vid_compressed_dir = '..\\data\\videos\\compressed'

training_config = struct()

training_config.number_of_epochs = 10
training_config.batch_size = 10
training_config.save_model_path = '../models/model.pth'
