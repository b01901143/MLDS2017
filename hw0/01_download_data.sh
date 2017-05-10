#!/bin/bash

echo "Download data and unzip them..."

wget -O data.zip https://www.dropbox.com/s/o9it4o2u1wgurdq/data.zip?dl=1
unzip -u-o data.zip
rm -rf data.zip

echo "Finish downloading data!!"
