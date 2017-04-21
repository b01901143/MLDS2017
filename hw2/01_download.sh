#!/bin/bash
echo "Download data and glove and unzip them..."

wget -O data.zip https://drive.google.com/a/media.ee.ntu.edu.tw/file/d/0B2hNk0_VowQmeEZoNENaWkIxX2c/view?usp=sharing
unzip -u-o data.zip
echo "Finish downloading data!!"

wget -O glove_42B.zip http://nlp.stanford.edu/data/glove.42B.300d.zip
unzip -u-o glove_42B.zip
echo "Finish downloading glove!!"
