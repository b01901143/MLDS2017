#!/bin/bash
echo "Generating model..."

model_version=$1
python generate.py $1

echo "Finish generating model!!"
