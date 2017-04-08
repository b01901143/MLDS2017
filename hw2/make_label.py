import os
import sys
import json
from nltk import word_tokenize

#argv
label_json_path = "./data/" + sys.argv[1] + "/" + sys.argv[1] + "_label.json"
label_dir = "./data/" + sys.argv[1] + "/label/"
sentence_length = sys.argv[2] #min_mutual_sentence_length = 9

#open
with open(label_json_path) as label_json_file:
	output_label = json.load(label_json_file)
	output_label_sorted = sorted(output_label, key=lambda k: k["id"])

#select
for video in output_label_sorted:
	temp_sentences = [ word_tokenize(sentence) for sentence in video["caption"] ]
	temp_strip_sentences = [ sentence for sentence in temp_sentences if(len(sentence) <= int(sentence_length)) ]
	temp_sorted_sentences = sorted(temp_strip_sentences, key=len)
	video["caption"] = temp_sorted_sentences[-1]

#mkdir
if not os.path.exists(label_dir):
	os.makedirs(label_dir)

#output
with open(label_dir + "label", "w") as label_file:
	json.dump(output_label_sorted, label_file)
