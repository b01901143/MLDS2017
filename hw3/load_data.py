import sys
import csv
import h5py
import argparse
import skipthoughts

#argparse
parser = argparse.ArgumentParser()
parser.add_argument("--images_dir_path", type=str, help="for example: ./data/basic/images/")
parser.add_argument("--text_file_path", type=str, help="for example: ./sample_training_text.txt")
parser.add_argument("--text_image_file_path", type=str, help="for example: ./text_image.hdf5")
args = parser.parse_args()

#prepare image_list and caption_list
image_list, caption_list = [], []
with open(args.text_file_path, "rb") as text_file:
	reader = csv.reader(text_file)
	for row in reader:
		image_list.append(row[0])
		caption_list.append(row[1])

#prepare out_dict
out_dict = {}
model = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)
for i, (image, caption) in enumerate(zip(image_list, caption_list)):
	out_dict[args.images_dir_path + image + ".jpg"] = encoder.encode([caption], verbose=False)
	sys.stdout.write("\rNow is at partition: {0: >8} / {1}".format(i + 1, len(image_list)))
	sys.stdout.flush()

#write
text_image_file = h5py.File(args.text_image_file_path)
for key in out_dict:
	text_image_file.create_dataset(key, data=out_dict[key])
text_image_file.close()
