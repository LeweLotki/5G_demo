#!/bin/bash

SOURCE_FILE="video_source_files_small.txt"
DEST_BASE_DIR="data_lj"
while IFS= read -r FILE_PATH; do
   echo "START    ======================="
   echo "$FILE_PATH"
   SUB_DIR=$(dirname "$FILE_PATH" | awk -F'original/' '{print $2}')
   echo "$SUB_DIR"
   FILE_NAME=$(basename "$FILE_PATH")
   echo "$FILE_NAME"
   DEST_DIR="$DEST_BASE_DIR/$SUB_DIR"
   echo "$DEST_DIR"
   mkdir -p "$DEST_DIR"
   DEST_FILE="$DEST_DIR/$FILE_NAME"
   echo "$DEST_FILE"
   
   ffmpeg -hide_banner -loglevel quiet -i $FILE_PATH -c:v libvpx-vp9 -b:v 1M -c:a libopus -b:a 128k $DEST_FILE > /dev/null 2>&1   
done < $SOURCE_FILE
