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

def readTextToImage(file_path):
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

def getBatchData(current_batch):
	real_image = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)
	caption = np.zeros((batch_size, caption_size), dtype=np.float32)
	image_file = []
	for i, (image_file_path, caption_array) in enumerate(current_batch):
		real_image[i,:,:,:] = getImageArray(image_file_path)
		caption[i,:] = caption_array.flatten()
		image_file.append(image_file_path)
	wrong_image = np.roll(real_image, 1, axis=0)
	noise = np.asarray(np.random.uniform(-1, 1, [batch_size, noise_size]), dtype=np.float32)
	return real_image, wrong_image, caption, noise, image_file

def getInferData(current_infer):
	caption = np.zeros((1, caption_size), dtype=np.float32)
	image_file = []
	for i, (image_file_path, caption_array) in enumerate(current_infer):
		caption[i,:] = caption_array.flatten()
		image_file.append(image_file_path)
	noise = np.asarray(np.random.uniform(-1, 1, [1, noise_size]), dtype=np.float32)
	return caption, noise, image_file

def saveImage(save_dir, saved_images):
	for i, arr in enumerate(saved_images):
		saved_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
		saved_image = arr
		scipy.misc.imsave(save_dir + str(i) + ".jpg", saved_image)

def sampleImage(sample_dir, sampled_images, batch_id):
	for arr in sampled_images:
		sampled_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
		sampled_image = arr
		scipy.misc.imsave(save_dir + batch_id + ".jpg", sampled_image)
