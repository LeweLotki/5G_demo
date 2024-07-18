import os
import torch
from identification import FacenetIdentifier
from identification import FrameExtractor


class Identification:

    dataset_size = 0
    def __init__(self):

        self.facenet = FacenetIdentifier()

    def run(self, faces: list, path: str) -> None:

        videos_dir = "../data/videos/original/all"
        output_dir = "../data/persons"
        
        extractor = FrameExtractor(videos_dir, output_dir)
        extractor.extract_frames()
           
        for face in faces:
            self.__add_face(face)

        self.__identify(path)

    def __add_face(self, path: str) -> None:
        self.facenet.add_face(path, f'Person{self.dataset_size}')
        self.dataset_size += 1
    
    def __identify(self, path: str) -> None: 
        self.facenet.train()
        self.facenet.identify_face(path)


