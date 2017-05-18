#!/bin/bash
echo "Prepare captions..."

INFO_DIR="info"
if [ ! -d "$INFO_DIR" ]; then
	mkdir -p "$INFO_DIR"
fi

infile_path="./data/basic/tags.csv"
outfile_path="./$INFO_DIR/sample_training_text.txt"
feature_type="hair eyes"

python prepare_captions.py --infile_path "$infile_path" --outfile_path "$outfile_path" --feature_type "$feature_type"

echo "Finish preparing captions!!"
