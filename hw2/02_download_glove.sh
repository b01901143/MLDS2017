#!/bin/bash
echo "Download glove and unzip them..."

wget -O glove.zip https://drive.google.com/a/media.ee.ntu.edu.tw/file/d/0B2hNk0_VowQmcFdhRUdwbFBlUXM/view?usp=sharing
unzip -u-o glove.zip

echo "Finish downloading glove!!"
