#!/bin/bash

testing_text=$1

sh 01_download_ski.sh
sh 02_download_models.sh
sh 05_sample_testing_info.sh

for model_version in 0 1 2 3 4
do
	sh 07_generate.sh $model_version
done
