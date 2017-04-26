#!/bin/bash
echo "Preprocess info..."

taken="one" #one/all
bound=10 #8~40

for type in "training" "testing"; do
	python preprocess.py $type $taken $bound
	echo "Finish Preprocessing" $type "info!!" 
done
