import sys
import csv
import json
import nltk
import codecs

#argv
label_json_path = "../data/" + sys.argv[1] + "/label.json"
label_dir = "../data/" + sys.argv[1] + "/label/"
info_csv_path = "../data/" + sys.argv[1] + "/info.csv"
label_taken = sys.argv[2] #one/all
length_upper_bound = sys.argv[3] #8~40

#open
output_label = json.load(codecs.open(label_json_path, "r", "utf-8"))

#preprocess
for label in output_label:
	temp_sentences = [ nltk.word_tokenize(sentence) for sentence in label["caption"] ]
	punct_word_list = [ ",", ".", "!", "?", " " ]
	temp_regex_sentences = [ [  word for word in sentence if word not in punct_word_list ] for sentence in temp_sentences ]
	temp_strip_sentences = [ sentence for sentence in temp_regex_sentences if(len(sentence) <= int(length_upper_bound)) ]
	temp_sort_sentences = sorted(temp_strip_sentences, key=len)
	temp_join_sentences = [ " ".join(sentence) for sentence in temp_sort_sentences ]
	with open(label_dir + label["id"] + ".json", "w") as label_file:
		if(label_taken == "all"):
			json.dump(temp_join_sentences, label_file, indent=4, sort_keys=True)
		elif(label_taken == "one"):
			json.dump([temp_join_sentences[-1]], label_file, indent=4, sort_keys=True)
		else:
			json.dump([temp_join_sentences[-1]], label_file, indent=4, sort_keys=True)			

#make_id_caption_dict
id_caption_list_of_list = [ [ label["id"] + ".npy", label["id"] + ".json" ] for label in output_label ]
id_caption_list_of_list.insert(0, ["feat_path", "label_path"])

#convert to utf-8
id_caption_list_of_list = [ [ unicode(w).encode("utf-8") for w in id_caption_list ] for id_caption_list in id_caption_list_of_list ]

#output
with open(info_csv_path, "w") as info_file:
	writer = csv.writer(info_file)
	writer.writerows(id_caption_list_of_list)
