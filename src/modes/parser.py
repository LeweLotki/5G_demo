from argparse import ArgumentParser
import argparse

from modes import (
    Training,
    Predict
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
        
        return parser

    def switch_mode(self):
       
        args = self.arg_parser.parse_args()

        if args.training: 
            Training()
            
        elif args.predict:
            predict = Predict(training_config.save_model_path)
            prediction = predict.predict(args.image_path)
            print(f'Prediction for given image is:\n{prediction}')

        else: 
            Training()

