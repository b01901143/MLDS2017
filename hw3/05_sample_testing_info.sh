#!/bin/bash
echo "Sample testing info..."

INFO_DIR="info"
if [ ! -d "$INFO_DIR" ]; then
	mkdir -p "$INFO_DIR"
fi

image_dir="./samples/"
text_file_path=$1
info_file_path="./$INFO_DIR/sample_testing_info"
python sample_testing_info.py --image_dir "$image_dir" --text_file_path "$text_file_path" --info_file_path "$info_file_path"

echo
echo "Finish sampling testing info!!"
