#!/bin/bash
echo "Sample testing info..."

INFO_DIR="info"

image_dir="./results/testing/"
text_file_path="./$INFO_DIR/sample_testing_text.txt"
info_file_path="./$INFO_DIR/sample_testing_info"
python sample_testing_info.py --image_dir "$image_dir" --text_file_path "$text_file_path" --info_file_path "$info_file_path"

echo
echo "Finish sampling testing info!!"
