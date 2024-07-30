import os
import re
import glob
import subprocess
import hashlib

# Define the source file containing the list of video files with full paths
SOURCE_FILE = "video_source_files.txt"

# Define the base directory for the new compressed files
DEST_BASE_DIR = "data_lj"
VIDEO_DIR = os.path.join(DEST_BASE_DIR, "videos")
FRAMES_DIR = os.path.join(DEST_BASE_DIR, "frames")

# Define the resolutions and bitrates
resolutions = ['384x216', '320x180', '240x135', '192x108']
bitrates = [
    [166, 83, 41],
    [115, 58, 29],
    [65, 32, 16],
    [41, 21, 10]
]

# Function to compress the video
def compress_video(input_file, output_file, resolution, bitrate):
    ffmpeg_command = [
        'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-i', input_file,
        '-vf', f'scale={resolution}', '-b:v', f'{bitrate}k', '-c:v', 'libvpx-vp9', '-c:a', 'libopus', '-b:a', '128k',
        output_file
    ]
    print(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}")

# Function to extract frames from the compressed file
def extract_frames(video_file, output_dir, base_name):
    os.makedirs(output_dir, exist_ok=True)
    ffmpeg_command = [
        'ffmpeg', '-i', video_file,
        os.path.join(output_dir, f'{base_name}_frame_%04d.png')
    ]
    print(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
    result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr.decode()}")

# Function to remove the first 30 and last 30 frames
def remove_unwanted_frames(output_dir, base_name):
    frames = sorted(glob.glob(os.path.join(output_dir, f'{base_name}_frame_*.png')))
    if len(frames) > 60:
        unwanted_frames = frames[:30] + frames[-30:]
        for frame in unwanted_frames:
            os.remove(frame)
    else:
        print(f"Not enough frames to remove from {output_dir}")

# Function to extract the number before 'x' and after '_'
def extract_number(filename):
    match = re.search(r'_(\d+)x', filename)
    if match:
        return match.group(1)
    return None

def generate_identifier(file_path):
    file_name = os.path.basename(file_path)
    return hashlib.md5(file_name.encode()).hexdigest()

# Function to process each file path
def process_file(file_path):
    print("START    =======================")
    print(file_path)

    # Generate a unique identifier for the video file
    identifier = generate_identifier(file_path)
    person_dir_name = f'person_{identifier}'
    
    # Get the subdirectory (th, th-bb, etc.) and file name
    sub_dir = os.path.dirname(file_path).split('original/', 1)[-1]
    print(sub_dir)

    file_name = os.path.basename(file_path)
    print(file_name)

    # Extract the first number before 'x' and after '_'
    number = extract_number(file_name)
    print(f"Extracted number: {number}")

    # Define the destination directory
    dest_dir = os.path.join(VIDEO_DIR, sub_dir, person_dir_name)
    os.makedirs(dest_dir, exist_ok=True)

    # Iterate through each resolution and its corresponding bitrates
    for res, brs in zip(resolutions, bitrates):
        # Reverse the resolution if the number is 1080
        if number == '1080':
            res_parts = res.split('x')
            res = f"{res_parts[1]}x{res_parts[0]}"

        for br in brs:
            # Define the destination file path for each resolution and bitrate
            dest_file = os.path.join(dest_dir, f"{file_name.rsplit('.', 1)[0]}_{res}_{br}.webm")
            print(dest_file)

            # Run the ffmpeg command to compress the video
            compress_video(file_path, dest_file, res, br)

            # Extract frames from the compressed file
            frame_output_dir = os.path.join(FRAMES_DIR, sub_dir, person_dir_name)
            base_name = f"{file_name.rsplit('.', 1)[0]}_{res}_{br}"
            extract_frames(dest_file, frame_output_dir, base_name)

            # Remove the first 30 and last 30 frames
            remove_unwanted_frames(frame_output_dir, base_name)

# Read each line from the source file
with open(SOURCE_FILE, 'r') as file:
    for line in file:
        file_path = line.strip()
        if file_path:
            process_file(file_path)

