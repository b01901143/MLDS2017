#!/bin/bash
echo "Prepare captions..."

infile_path="./data/basic/tags.csv"
outfile_path="./sample_training_text.txt"
feature_type="hair eyes"

python prepare_captions.py --infile_path "$infile_path" --outfile_path "$outfile_path" --feature_type "$feature_type"

echo "Finish preparing captions!!"
