#!/bin/bash

FOLDER_ID="1SZ9A1GnvdmLDJ09PV-6IszCNDZs4IpkQ" 

OUTPUT_DIR="generated_images"

if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

gdown --folder "https://drive.google.com/drive/folders/$FOLDER_ID" -O "$OUTPUT_DIR"

echo "Download complete. Files saved in: $OUTPUT_DIR"