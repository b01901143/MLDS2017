import sys
import csv
import pickle
import argparse
import skipthoughts

#argparse
parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, help="for example: ./results/testing/")
parser.add_argument("--text_file_path", type=str, help="for example: ./info/sample_testing_text.txt")
parser.add_argument("--info_file_path", type=str, help="for example: ./info/sample_testing_info")
args = parser.parse_args()

#prepare image_list and caption_list
image_list, caption_list = [], []
with open(args.text_file_path, "rb") as text_file:
	reader = csv.reader(text_file)
	for row in reader:
		image_list.append(row[0])
		caption_list.append(row[1])

#prepare out_dict
out_list = []
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
for i, (image, caption) in enumerate(zip(image_list, caption_list)):
	out_list.append( ( 
			image,
			args.image_dir + image + ".jpg",
			caption, 
			encoder.encode([caption], verbose=False),
		) 
	)
	sys.stdout.write("\rNow is at partition: {0: >8} / {1}".format(i + 1, len(image_list)))
	sys.stdout.flush()

#write
with open(args.info_file_path, "wb") as info_file:
	pickle.dump(out_list, info_file, protocol=pickle.HIGHEST_PROTOCOL)
