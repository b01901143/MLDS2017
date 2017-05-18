#!/bin/bash
echo "Load train data..."

INFO_DIR="info"

images_dir_path="./data/basic/images/"
text_file_path="./$INFO_DIR/sample_training_text.txt"
text_image_file_path="./$INFO_DIR/training_text_image"
python load_data.py --images_dir_path "$images_dir_path" --text_file_path "$text_file_path" --text_image_file_path "$text_image_file_path"

echo
echo "Finish loading train data!!"
