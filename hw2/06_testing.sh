#!/bin/bash
echo "Test VCG model..."

model="D" #D/D_ATTEN/DES/DEJ four types of models
path=$model"/"test.py

python $path 

echo "Finish testing!!"
