import os
import sys
import json
from nltk import sent_tokenize

#argv
label_json_path = sys.argv[1]
labels_dir = sys.argv[2]
sentence_length = sys.argv[3] #[""]

#open
with open(label_json_path) as label_json_file:
	label_data = json.load(label_json_file)

#mkdir
if os.path.exists(labels_dir):
	os.makedirs(labels_dir)

#get captions, ids
for video in label_data:
	sentence_list = []
	for sentence in video["caption"]:
		temp = sent_tokenize(sentence)
		sentence_list.append(temp)
		
