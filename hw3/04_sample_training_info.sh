#!/bin/bash
echo "Sample training info..."

INFO_DIR="info"
if [ ! -d "$INFO_DIR" ]; then
	mkdir -p "$INFO_DIR"
fi

image_dir="./data/basic/images/"
text_file_path="./$INFO_DIR/sample_training_text.txt"
info_file_path="./$INFO_DIR/sample_training_info"
python sample_training_info.py --image_dir "$image_dir" --text_file_path "$text_file_path" --info_file_path "$info_file_path"

echo
echo "Finish sampling training info!!"
