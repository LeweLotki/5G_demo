import os
import subprocess

class Compressor:
    def __init__(self, input_dir, output_dir, ffmpeg_path="ffmpeg"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.ffmpeg_path = ffmpeg_path
        self.supported_formats = (".mp4", ".avi", ".mov")  # Add other video formats as needed

    def compress_videos(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(self.supported_formats):
                    self.compress_video(root, file)

    def compress_video(self, root, file):
        input_path = os.path.join(root, file)
        output_path = self.get_output_path(root, file)
        output_subdir = os.path.dirname(output_path)
        
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        if os.path.exists(output_path):
            # print(f"Skipping {output_path}, already exists.")
            return
        
        self.run_ffmpeg_command(input_path, output_path)

    def get_output_path(self, root, file):
        relative_path = os.path.relpath(root, self.input_dir)
        return os.path.join(self.output_dir, relative_path, file)

    def run_ffmpeg_command(self, input_path, output_path):
        command = [
            self.ffmpeg_path,
            "-i", input_path,
            "-vcodec", "libx264",
            "-crf", "28",
            output_path
        ]
        subprocess.run(command)

# Usage example
if __name__ == "__main__":
    input_directory = "./data/videos/original"
    output_directory = "./data/videos/compressed"
    compressor = Compressor(input_directory, output_directory)
    compressor.compress_videos()
