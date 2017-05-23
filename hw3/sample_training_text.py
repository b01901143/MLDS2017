import csv
import argparse

#argparse
parser = argparse.ArgumentParser()
parser.add_argument("--infile_path", type=str, help="for example: ./data/basic/tags.csv")
parser.add_argument("--outfile_path", type=str, help="for example: ./info/sample_training_text.txt")
parser.add_argument("--feature_type", type=str, help="for example: \"all\" or \"hair, eyes\"")
args = parser.parse_args()

#read
id_list, caption_list = [], []
with open(args.infile_path, "rb") as infile:
	reader = csv.reader(infile)
	for row in reader:
		id_list.append(row[0])
		captions = row[1].split("\t")
		temp_tuple_list = [ ( caption.split(":")[0], int(caption.split(":")[1]) ) for caption in captions if caption is not "" ]
		temp_sorted_tuple_list = sorted(temp_tuple_list, key=lambda x: x[1], reverse=True)
		if(args.feature_type == "all"):
			caption_list.append(" ".join([ t[0] for t in temp_sorted_tuple_list ]))
		else:
			caption_list.append(" ".join([ t[0] for key_word in args.feature_type.split(" ") for t in temp_sorted_tuple_list if key_word in t[0] ]))

with open(args.outfile_path, "wb") as outfile:
	writer = csv.writer(outfile)
	for id_, caption in zip(id_list, caption_list):
		if caption is not "":
			writer.writerow([id_, caption])
