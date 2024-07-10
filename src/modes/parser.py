from argparse import ArgumentParser
import argparse

from modes import (
    Training,
    Predict,
    Detect
)

from config import (
    dataset_config,
    training_config
)

class Parser:

    def __init__(self):
        self.arg_parser = self.__setup_arg_parser()

    def __setup_arg_parser(self):
        parser = ArgumentParser()
        training_parser = parser.add_argument_group('Training Mode')
        training_parser.add_argument('-t','--training', action='store_true')

        predict_parser = parser.add_argument_group('Predict Mode')
        predict_parser.add_argument('-p', '--predict', action='store_true')
        predict_parser.add_argument('--image_path', type=str, required=False, help='Path to the input image')

        detect_parser = parser.add_argument_group('Detect Mode')
        detect_parser.add_argument('-d', '--detect', action='store_true')
        detect_parser.add_argument('--dir_path', type=str, required=False, help='Path to the directory containing images')
        detect_parser.add_argument('--threshold', type=float, required=False, help='Threshold of detection')

        return parser

    def switch_mode(self):
        args = self.arg_parser.parse_args()

        if args.training:
            Training()
        elif args.predict:
            predict = Predict(training_config.save_model_path)
            prediction = predict.predict(args.image_path)
            print(f'Prediction for given image is:\n{prediction}')
        elif args.detect:
            detect = Detect()
            if args.dir_path:
                detect.detect_in_directory(dir_path=args.dir_path, threshold=args.threshold)
            elif args.image_path:
                detect.detect(image_path=args.image_path, threshold=args.threshold)
            else:
                print('Please provide either an image path or a directory path for detection.')
        else:
            Training()

