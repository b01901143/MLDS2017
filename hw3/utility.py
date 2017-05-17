import random
import pickle
import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.transform
import numpy as np
from parameter import *

def convertDictToList(dict_file_path):
	with open(dict_file_path, "rb") as dict_file:
		dict_ = pickle.load(dict_file)
	list_ = [ (key, value) for key, value in dict_.iteritems() ]
	return list_

def getImageArray(image_file_path):
	image_array = skimage.io.imread(image_file_path)
	resized_image_array = skimage.transform.resize(image_array, (image_size, image_size))
	if random.random() > 0.5:
		resized_image_array = np.fliplr(resized_image_array)
	return resized_image_array.astype(np.float32)

def getBatchData(current_batch):
	real_images = np.zeros((batch_size, image_size, image_size, 3), dtype=np.float32)
	captions = np.zeros((batch_size, caption_size), dtype=np.float32)
	image_files = []
	for i, (image_file_path, caption_array) in enumerate(current_batch):
		real_images[i,:,:,:] = getImageArray(image_file_path)
		captions[i,:] = caption_array.flatten()
		image_files.append(image_file_path)
	wrong_images = np.roll(real_images, 1, axis=0)
	z_noise = np.asarray(np.random.uniform(-1, 1, [batch_size, z_dim]), dtype=np.float32)
	return real_images, wrong_images, captions, z_noise, image_files

def saveGenImage(save_dir, generated_images):
	for i, arr in enumerate(generated_images):
		generated_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
		generated_image = arr
		scipy.misc.imsave(save_dir + str(i) + ".jpg", generated_image)
