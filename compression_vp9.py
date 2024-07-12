import os
import subprocess
import re

# Define the source file containing the list of video files with full paths
SOURCE_FILE = "video_source_files_small.txt"

# Define the base directory for the new compressed files
DEST_BASE_DIR = "data_lj"

# Define the resolutions and bitrates
resolutions = ['384x216', '320x180', '240x135', '192x108']
bitrates = [
    [166, 83, 41],
    [115, 58, 29],
    [65, 32, 16],
    [41, 21, 10]
]

# Function to process each file path
def process_file(file_path):
    print("START    =======================")
    print(file_path)
    
    # Get the subdirectory (th, th-bb, etc.) and file name
    sub_dir = os.path.dirname(file_path).split('original/', 1)[-1]
    print(sub_dir)
    
    file_name = os.path.basename(file_path)
    print(file_name)

    # Extract the first number before 'x' and after '_'
    number = extract_number(file_name)
    print(f"Extracted number: {number}")

    # Define the destination directory
    dest_dir = os.path.join(DEST_BASE_DIR, sub_dir)
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

            # Run the ffmpeg command with suppressed output and overwrite option
            ffmpeg_command = [
                'ffmpeg', '-y', '-hide_banner', '-loglevel', 'quiet', '-i', file_path,
                '-vf', f'scale={res}', '-b:v', f'{br}k', '-c:v', 'libvpx-vp9', '-c:a', 'libopus', '-b:a', '128k', dest_file
            ]
            subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

# Function to extract the number before 'x' and after '_'
def extract_number(filename):
    # Use regex to find the pattern that matches the number before 'x' and after '_'
    match = re.search(r'_(\d+)x', filename)
    if match:
        return match.group(1)
    return None

# Read each line from the source file
with open(SOURCE_FILE, 'r') as file:
    for line in file:
        file_path = line.strip()
        if file_path:
            process_file(file_path)