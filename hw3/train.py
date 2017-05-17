import os
import pickle
import random
import scipy
import scipy.misc
import skimage
import skimage.io
import skimage.transform
import model
import numpy as np
import tensorflow as tf

image_size = 64
caption_size = 4800
z_dim = 100
t_dim = 256
gf_dim = 64
df_dim = 64
gfc_dim = 1024
num_epoch = 600
save_num_batch = 30
restore_flag = None
batch_size = 64
learning_rate = 0.0002
beta1 = 0.5

save_dir = "./saves/"
text_image_path = "./text_image"

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

def train():
	text_image_list = convertDictToList(text_image_path)
	model_options = {
		'z_dim' : z_dim,
		't_dim' : t_dim,
		'batch_size' : batch_size,
		'image_size' : image_size,
		'gf_dim' : gf_dim,
		'df_dim' : df_dim,
		'gfc_dim' : gfc_dim,
		'caption_vector_length' : caption_size
	}
	GAN = model.GAN(model_options)
	with tf.variable_scope("GAN"):
		input_tensors, variables, loss, outputs, checks = GAN.build_model()
	d_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss['d_loss'], var_list=variables['d_vars'])
	g_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(loss['g_loss'], var_list=variables['g_vars'])
	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	tf.initialize_all_variables().run()
	if restore_flag:
		saver.restore(session, restore_flag)
	for epoch in range(num_epoch):
		random.shuffle(text_image_list)
		for batch in range(len(text_image_list)//batch_size):
			current_batch = text_image_list[batch*batch_size:(batch+1)*batch_size]
			real_images, wrong_images, captions, z_noise, image_files = getBatchData(current_batch)
			_, d_loss, gen, d1, d2, d3 = session.run( [ d_optimizer, loss['d_loss'], outputs['generator'] ] + [ checks['d_loss1'] , checks['d_loss2'], checks['d_loss3'] ],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : captions,
					input_tensors['t_z'] : z_noise,
				}
			)
			_, g_loss, gen = session.run( [ g_optimizer, loss['g_loss'], outputs['generator'] ],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : captions,
					input_tensors['t_z'] : z_noise,
				}
			)
			_, g_loss, gen = session.run( [ g_optimizer, loss['g_loss'], outputs['generator'] ],
				feed_dict = {
					input_tensors['t_real_image'] : real_images,
					input_tensors['t_wrong_image'] : wrong_images,
					input_tensors['t_real_caption'] : captions,
					input_tensors['t_z'] : z_noise,
				}
			)
			print "Losses: ", d_loss, g_loss, "Batch: ", batch
			if (batch % save_num_batch) == 0:
				print "Saving images..."
				if not os.path.exists(save_dir):
					os.makedirs(save_dir)
				saveGenImage(save_dir, gen)
		save_path = saver.save(session, "models/epoch_{0}.ckpt".format(epoch))

if __name__ == '__main__':
	train()
