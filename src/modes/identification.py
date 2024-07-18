import os
import torch
from identification import FacenetIdentifier
from identification import FrameExtractor


class Identification:
    
    def __init__(self):
        self.facenet = FacenetIdentifier()

    def run(self, path: str) -> None:
        videos_dir = "../data/videos/original/all"
        output_dir = "../data/persons"
        
        extractor = FrameExtractor(videos_dir, output_dir)
        extractor.extract_frames()
        
        self.load_and_add_faces(output_dir)
           
        self.__identify(path)

    def __add_face(self, path: str, label: str) -> None:
        self.facenet.add_face(path, label)
    
    def __identify(self, path: str) -> None: 
        self.facenet.train()
        self.facenet.identify_face(path)

    def load_and_add_faces(self, base_dir: str) -> None:
        for person_dir in os.listdir(base_dir):
            person_path = os.path.join(base_dir, person_dir)
            if os.path.isdir(person_path):
                for frame in os.listdir(person_path):
                    frame_path = os.path.join(person_path, frame)
                    if frame_path.endswith('.jpg'):
                        self.__add_face(frame_path, person_dir)

