#!/bin/bash
echo "Load test data..."

INFO_DIR="info"

images_dir_path="./samples/"
text_file_path="./$INFO_DIR/sample_testing_text.txt"
text_image_file_path="./$INFO_DIR/testing_text_image"
python load_data.py --images_dir_path "$images_dir_path" --text_file_path "$text_file_path" --text_image_file_path "$text_image_file_path"

echo
echo "Finish loading test data!!"
