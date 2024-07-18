#!/bin/bash

# Define the source file containing the list of video files with full paths
SOURCE_FILE="video_source_files_small.txt"
echo "1"
# Define the base directory for the new compressed files
DEST_BASE_DIR="data_lj"
echo "2"
# Read each line from the source file
while IFS= read -r FILE_PATH; do
    # Get the subdirectory (th, th-bb, etc.) and file name
    echo "3"
    echo "$FILE_PATH"
    SUB_DIR=$(dirname "$FILE_PATH" | awk -F'original/' '{print $2}')
    echo "$SUB_DIR"
    FILE_NAME=$(basename "$FILE_PATH")

    # Define the destination directory and file path
    DEST_DIR="$DEST_BASE_DIR/$SUB_DIR"
    mkdir -p "$DEST_DIR"

    DEST_FILE="$DEST_DIR/$FILE_NAME"

    echo "Running: ffmpeg -hide_banner -loglevel quiet -i \"$FILE_PATH\" -c:v libvpx-vp9 -b:v 1M -c:a libopus -b:a 128k \"$DEST_FILE\""

    ffmpeg -hide_banner -loglevel quiet -i "$FILE_PATH" -c:v libvpx-vp9 -b:v 1M -c:a libopus -b:a 128k "$DEST_FILE" > /dev/null 2>&1
    
done < "$SOURCE_FILE"

