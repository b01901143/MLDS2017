import sys
import csv
import json
import nltk
import codecs

#argv
label_json_path = "./data/" + sys.argv[1] + "/label.json"
label_dir = "./data/" + sys.argv[1] + "/label/"
info_csv_path = "./data/" + sys.argv[1] + "/info.csv"
sentence_length = sys.argv[2] #min_mutual_sentence_length = 8, max_mutual_sentence_length = 40

#open
output_label = json.loada(codecs.open(label_json_path, "r", "utf-8"))

#preprocess
for label in output_label:
	temp_sentences = [ nltk.word_tokenize(sentence) for sentence in label["caption"] ]
	punct_word_list = [ ",", ".", "!", "?", " " ]
	temp_regex_sentences = [ [  word for word in sentence if word not in punct_word_list ] for sentence in temp_sentences ]
	temp_strip_sentences = [ sentence for sentence in temp_regex_sentences if(len(sentence) <= int(sentence_length)) ]
	temp_join_sentences = [ " ".join(sentence) for sentence in temp_strip_sentences ]
	with open(label_dir + label["id"] + ".json", "w") as label_file:
		json.dump(temp_join_sentences, label_file, indent=4, sort_keys=True) 

#make_id_caption_dict
id_caption_list_of_list = [ [ label["id"] + ".npy", label["id"] + ".json" ] for label in output_label ]
id_caption_list_of_list.insert(0, ["feat_path", "label_path"])

#convert to utf-8
id_caption_list_of_list = [ [ unicode(w).encode("utf-8") for w in id_caption_list ] for id_caption_list in id_caption_list_of_list ]

#output
with open(info_csv_path, "w") as info_file:
	writer = csv.writer(info_file)
	writer.writerows(id_caption_list_of_list)
