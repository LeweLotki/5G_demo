import os
import random
import torch
from identification import FacenetIdentifier
from identification import FrameExtractor, TestFrameExtractor

class Identification:
    
    def __init__(self):
        self.facenet = FacenetIdentifier()

    def identify_image(self, path: str) -> None:
        videos_dir = "../data/videos/original/all"
        output_dir = "../data/persons"
        test_output_dir = "../data/persons/test"

        extractor = FrameExtractor(
            videos_dir=videos_dir, 
            output_dir=output_dir,
            frame_count=4
        )
        extractor.extract_frames()
        
        test_extractor = TestFrameExtractor(
            videos_dir=videos_dir, 
            output_dir=output_dir, 
            test_output_dir=test_output_dir
        )
        test_extractor.extract_test_frames()

        self.__load_and_add_faces(output_dir)
           
        self.__identify(path)

    def identify_test_set(self) -> None:
        videos_dir = "../data/videos/original/all"
        output_dir = "../data/persons"
        test_output_dir = "../data/persons/test"
        compressed_dir = "../data_lj/frames"

        extractor = FrameExtractor(
            videos_dir=videos_dir, 
            output_dir=output_dir,
            frame_count=4
        )
        extractor.extract_frames()
        
        test_extractor = TestFrameExtractor(
            videos_dir=videos_dir, 
            output_dir=output_dir, 
            test_output_dir=test_output_dir
        )
        test_extractor.extract_test_frames()

        self.__load_and_add_faces(output_dir)
        
        self.__calculate_accuracy(compressed_dir)

    def __calculate_accuracy(self, test_output_dir, num_frames=100):
        correct_identifications = 0
        total_frames = 0

        print(f"Starting accuracy calculation in directory: {test_output_dir}")

        try:
            for sub_dir in os.listdir(test_output_dir):
                sub_dir_path = os.path.join(test_output_dir, sub_dir)
                print(f"Checking sub directory: {sub_dir_path}")
                if os.path.isdir(sub_dir_path):
                    for person_dir in os.listdir(sub_dir_path):
                        person_path = os.path.join(sub_dir_path, person_dir)
                        print(f"Checking person directory: {person_path}")
                        if os.path.isdir(person_path):
                            frame_files = [f for f in os.listdir(person_path) if f.endswith('.jpg') or f.endswith('.png')]
                            if len(frame_files) > num_frames:
                                frame_files = random.sample(frame_files, num_frames)
                            for frame in frame_files:
                                frame_path = os.path.join(person_path, frame)
                                print(f"Checking frame file: {frame_path}")
                                identified_person, identification_probability = self.__identify(frame_path)
                                print(f'face: {person_dir} identified as face: {identified_person} with probability: {identification_probability}')
                                if identified_person == person_dir:
                                    correct_identifications += 1
                                total_frames += 1

        finally:
            if total_frames > 0:
                accuracy = correct_identifications / total_frames
                print(f"Accuracy: {accuracy:.2%}")
            else:
                print("No frames to evaluate accuracy.")

    def __add_face(self, path: str, label: str) -> None:
        self.facenet.add_face(path, label)
    
    def __identify(self, path: str) -> None: 
        self.facenet.train()
        identified_person, identification_probability = self.facenet.identify_face(path)
        return identified_person, identification_probability

    def __load_and_add_faces(self, base_dir: str) -> None:
        for person_dir in os.listdir(base_dir):
            person_path = os.path.join(base_dir, person_dir)
            if os.path.isdir(person_path):
                for frame in os.listdir(person_path):
                    frame_path = os.path.join(person_path, frame)
                    if frame_path.endswith('.jpg'):
                        self.__add_face(frame_path, person_dir)

