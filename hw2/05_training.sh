#!/bin/bash
echo "Train VCG model..."

model="D" #D/D_ATTEN/DES/DEJ four types of models
path=$model"/"train.py

python $path 

echo "Finish training!!"
