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

def readSampleInfo(file_path):
	with open(file_path, "rb") as file:
		list_ = pickle.load(file)
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
	for i, sample_data in enumerate(train_data):
		real_image[i,:,:,:] = getImageArray(sample_data[1])
		caption[i,:] = sample_data[3].flatten()
		image_file.append(sample_data[1])
	wrong_image = np.roll(real_image, 1, axis=0)
	noise = np.asarray(np.random.uniform(-1, 1, [batch_size, noise_size]), dtype=np.float32)
	return real_image, wrong_image, caption, noise, image_file

def getTestData(test_data):
	caption = np.zeros((1, caption_size), dtype=np.float32)
	image_file = []
	for i, sample_data in enumerate(test_data):
		caption[i,:] = sample_data[3].flatten()
		image_file.append(sample_data[1])
	noise = np.asarray(np.random.uniform(-1, 1, [1, noise_size]), dtype=np.float32)
	return caption, noise, image_file
