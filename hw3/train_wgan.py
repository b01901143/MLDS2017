import os
import sys
import random
import tensorflow as tf
from utility import *
from parameter import *
from model_wgan import *

def train():
	#get text_image info
	sample_training_info_list = readSampleInfo(sample_training_info_path)
	#model
	model = GAN(
		image_size=image_size,
		caption_size=caption_size,
		embedding_size=embedding_size,
		noise_size=noise_size,
		g_channel_size=g_channel_size,
		d_channel_size=d_channel_size,
		batch_size=batch_size
	)
	#build train model
	with tf.variable_scope("GAN"):
		input_tensor, output_tensor, loss_tensor, variable_tensor, check_tensor = model.buildTrainModel()
	#optimizer
	g_optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss_tensor['g_loss'], var_list=variable_tensor['g_variable'])
	d_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_tensor['d_loss'], var_list=variable_tensor['d_variable'])
	if d_optimizer >= 0.01:
		d_optimizer = 0.01
	if d_optimizer <= -0.01:
		d_optimizer = -0.01 
	#session and saver
	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	if restore_flag:
		saver.restore(session, model_dir + "-" + str(restore_version))
	else:
		session.run(tf.global_variables_initializer())
	#makedirs
	if not os.path.exists(model_dir):
		os.makedirs(model_dir)
	if not os.path.exists(result_training_dir):
		os.makedirs(result_training_dir)
	#epoch
	for epoch in range(num_epoch):
		random.shuffle(sample_training_info_list)
		for batch in range(len(sample_training_info_list) // batch_size):
			current_train_data = sample_training_info_list[batch * batch_size : (batch+1) * batch_size]
			real_image, wrong_image, caption, noise, image_file = getTrainData(current_train_data)
			feed_dict = {
				input_tensor["real_image"]: real_image,
				input_tensor["wrong_image"]: wrong_image,
				input_tensor["caption"]: caption,
				input_tensor["noise"]: noise
			}			
			g_fetch_dict = {
				"g_optimizer": g_optimizer,
				"g_loss": loss_tensor["g_loss"],
				"fake_image": output_tensor["fake_image"]
			}
			d_fetch_dict = {
				"d_optimizer": d_optimizer,
				"d_loss": loss_tensor["d_loss"],
				"fake_image": output_tensor["fake_image"],
				"d_loss_1": check_tensor["d_loss_1"],
				"d_loss_2": check_tensor["d_loss_2"],
				"d_loss_3": check_tensor["d_loss_3"]
			}
			d_track_dict = session.run(d_fetch_dict, feed_dict=feed_dict)
			g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
			g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
			sys.stdout.write("\rBatchID: {0}, G losses: {1}, D losses: {2}".format(batch, g_track_dict["g_loss"], d_track_dict["d_loss"]))
			sys.stdout.flush()
			if (batch % save_num_batch) == 0:
				for f in os.listdir(result_training_dir):
					file_path = os.path.join(result_training_dir, f)
					os.unlink(file_path)
				with open(result_caption_path, "wb") as result_training_caption_file:
					writer = csv.writer(result_training_caption_file)	
					list_ = []
					for i, (image, file) in enumerate(zip(g_track_dict["fake_image"], image_file)):
						saved_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
						saved_image = image
						id_ = file.split("/")[-1].split(".")[0]
						caption = sample_training_text_dict[id_]
						list_.append((int(id_), (caption, saved_image)))
					sorted(list_, key=lambda x: list_[0])
					for id_, (caption, saved_image) in list_:
						scipy.misc.imsave(result_dir + str(id_) + ".jpg", saved_image)
						writer.writerow([id_, caption])	
		sys.stdout.write("\nEpochID: {0}, Saving Model...\n".format(epoch))
		saver.save(session, model_dir, global_step=epoch)
		if (epoch % save_num_epoch) == 0 :
			saver.save(session, model_dir, global_step=epoch)

if __name__ == '__main__':
	train()
