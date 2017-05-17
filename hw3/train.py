import os
import random
import tensorflow as tf
from parameter import *
from utility import *
import model

def train():
	text_image_list = convertDictToList(text_image_path)
	GAN = model.GAN()
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
