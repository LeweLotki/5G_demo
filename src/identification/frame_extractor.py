import cv2
import os
import random

class FrameExtractor:
    def __init__(self, videos_dir, output_dir):
        self.videos_dir = videos_dir
        self.output_dir = output_dir
        self.frame_count = 4

    def extract_frames(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        video_files = [f for f in os.listdir(self.videos_dir) if f.endswith('.mp4')]
        
        for idx, video_file in enumerate(video_files):
            person_dir = os.path.join(self.output_dir, f'person{idx}')
            if not os.path.exists(person_dir):
                os.makedirs(person_dir)
            
            frame_files = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            if len(frame_files) >= self.frame_count:
                continue
            
            self.process_video(video_file, person_dir)

    def process_video(self, video_file, person_dir):
        video_path = os.path.join(self.videos_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error opening video file {video_file}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        first_second_frames = min(fps, total_frames)
        last_second_frames = min(fps, total_frames - (total_frames - fps))

        first_second_indices = random.sample(range(first_second_frames), 2)
        last_second_indices = random.sample(range(total_frames - fps, total_frames), 2)

        self.save_frames(cap, person_dir, first_second_indices, "first")
        self.save_frames(cap, person_dir, last_second_indices, "last")

        cap.release()

    def save_frames(self, cap, person_dir, frame_indices, prefix):
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame_path = os.path.join(person_dir, f'{prefix}_{i}.jpg')
                cv2.imwrite(frame_path, frame)
            else:
                print(f"Failed to read frame {frame_idx}.")

if __name__ == "__main__":
    videos_dir = "../data/videos/original"
    output_dir = "../data/persons"
    
    extractor = FrameExtractor(videos_dir, output_dir)
    extractor.extract_frames()

