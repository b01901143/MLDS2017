#!/bin/bash
echo "Download glove and unzip them..."

curl -H "Authorization: Bearer ya29.GltHBFwtUhu7ug_H81RKrVQPEw6al0iOZSEZ-rHYrRd4buTcxB-mHFw2AYtFnR6ngszIoENEAJrHD08tNkZ7c8xyplOjWaC6tfhCJACDwl9PDmjHeJzN2kB1vFJh" https://www.googleapis.com/drive/v3/files/0B2hNk0_VowQmcFdhRUdwbFBlUXM?alt=media -o glove.zip
unzip -u-o glove.zip
rm -rf glove.zip

echo "Finish downloading glove!!"
