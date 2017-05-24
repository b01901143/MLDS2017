import csv
import random
import pickle
import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.transform
import numpy as np
from parameter import *

def readSampleText(file_path):
	with open(file_path, "rb") as file:
		reader = csv.reader(file)
		dict_ = { row[0]: row[1] for row in reader }
		return dict_

def readSampleInfo(file_path):
	with open(file_path, "rb") as file:
		dict_ = pickle.load(file)
	list_ = [ (key, value) for key, value in dict_.iteritems() ]
	return list_

def getImageArray(image_file_path):
	image_array = skimage.io.imread(image_file_path)
	resized_image_array = skimage.transform.resize(image_array, (image_size, image_size))
	if random.random() > 0.5:
		resized_image_array = np.fliplr(resized_image_array)
	return resized_image_array.astype(np.float32)

def getTrainData(train_data):
	real_image = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)
	caption = np.zeros((batch_size, caption_size), dtype=np.float32)
	image_file = []
	for i, (image_file_path, caption_array) in enumerate(train_data):
		real_image[i,:,:,:] = getImageArray(image_file_path)
		caption[i,:] = caption_array.flatten()
		image_file.append(image_file_path)
	wrong_image = np.roll(real_image, 1, axis=0)
	noise = np.asarray(np.random.uniform(-1, 1, [batch_size, noise_size]), dtype=np.float32)
	return real_image, wrong_image, caption, noise, image_file

def getTestData(test_data):
	caption = np.zeros((1, caption_size), dtype=np.float32)
	image_file = []
	for i, (image_file_path, caption_array) in enumerate(test_data):
		caption[i,:] = caption_array.flatten()
		image_file.append(image_file_path)
	noise = np.asarray(np.random.uniform(-7, 7, [1, noise_size]), dtype=np.float32)
	return caption, noise, image_file

def saveImageCaption(result_dir, result_caption_path, sample_training_text_dict, fake_image, image_file):
	with open(result_caption_path, "wb") as result_training_caption_file:
		writer = csv.writer(result_training_caption_file)	
		list_ = []
		for i, (image, file) in enumerate(zip(fake_image, image_file)):
			saved_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
			saved_image = image
			id_ = file.split("/")[-1].split(".")[0]
			caption = sample_training_text_dict[id_]
			list_.append((int(id_), (caption, saved_image)))
		sorted(list_, key=lambda x: list_[0])
		for id_, (caption, saved_image) in list_:
			scipy.misc.imsave(result_dir + str(id_) + ".jpg", saved_image)
			writer.writerow([id_, caption])