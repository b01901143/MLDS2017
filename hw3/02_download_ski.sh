#!/bin/bash
echo "Download ski..."

SKI_DIR="ski"
if [ ! -d "$SKI_DIR" ]; then
	mkdir -p "$SKI_DIR"
fi

for name in dictionary.txt utable.npy btable.npy uni_skip.npz uni_skip.npz.pkl bi_skip.npz bi_skip.npz.pkl; do
	wget http://www.cs.toronto.edu/~rkiros/models/"$name" -P "$SKI_DIR"
done

echo "Finish downloading ski!!"
