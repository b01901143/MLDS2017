import os
import sys
import csv
import json
import codecs

#argv
label_json_path = "./data/" + sys.argv[1] + "/" + sys.argv[1] + "_label.json"
label_dir = "./data/" + sys.argv[1] + "/label/"
sentence_length = sys.argv[2] #min_mutual_sentence_length = 9

#open
output_label = json.load(codecs.open(label_json_path, "r", "utf-8"))

#select
for label in output_label:
	temp_sentences = [ sentence.split(" ") for sentence in label["caption"] ]
	temp_strip_sentences = [ sentence for sentence in temp_sentences if(len(sentence) <= int(sentence_length)) ]
	temp_sorted_sentences = sorted(temp_strip_sentences, key=len)
	label["caption"] = temp_sorted_sentences[-1]

#make_id_caption_dict
id_caption_list_of_list = [ [ label["id"], " ".join(label["caption"]) ] for label in output_label ]
id_caption_list_of_list.insert(0, ["VideoID", "Description"])

#convert to utf-8
id_caption_list_of_list = [ [ unicode(w).encode("utf-8") for w in id_caption_list ] for id_caption_list in id_caption_list_of_list ]

#mkdir
if not os.path.exists(label_dir):
	os.makedirs(label_dir)

#output
with open(label_dir + "/final_label.csv", "w") as label_file:
	writer = csv.writer(label_file)
	writer.writerows(id_caption_list_of_list)
