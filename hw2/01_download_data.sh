#!/bin/bash
echo "Download data and unzip them..."

wget -O data.zip https://drive.google.com/a/media.ee.ntu.edu.tw/file/d/0B2hNk0_VowQmeEZoNENaWkIxX2c/view?usp=sharing
unzip -u-o data.zip

echo "Finish downloading data!!"
