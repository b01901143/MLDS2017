import os
import sys
import scipy
from utility import *
from parameter import *
from model import *

sample_num = 5
sample_dir = "samples/"
def generate():
	#get text_image info
	sample_testing_text_dict = readSampleText(sample_testing_text_path)
	sample_testing_info_list = readSampleInfo(sample_testing_info_path)
	#model
	model = GAN(
		image_size=image_size,
		caption_size=caption_size,
		embedding_size=embedding_size,
		noise_size=noise_size,
		g_channel_size=g_channel_size,
		d_channel_size=d_channel_size,
		batch_size=1
	)	
	#build test model
	with tf.variable_scope("GAN"):
		input_tensor, output_tensor = model.buildTestModel()
	#session and saver
	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(session, model_dir + "-" + str(restore_version))
	#makedirs
	if not os.path.exists(result_testing_dir):
		os.makedirs(result_testing_dir)
	for batch in range(len(sample_testing_info_list) // 1):
		for i in range(sample_num):
			current_test_data = sample_testing_info_list[batch : (batch+1) * 1]
			caption, noise, image_file = getTestData(current_test_data)
			feed_dict = {
				input_tensor['caption']: caption,
				input_tensor['noise']: noise
			}
			g_fetch_dict = {
				"fake_image": output_tensor["fake_image"]
			}
			g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
			sys.stdout.write("\rBatchID: {0}, Saving Image...".format(batch))
			sys.stdout.flush()
			saveImageCaption(result_testing_dir, result_testing_caption_path, sample_testing_text_dict, g_track_dict["fake_image"], image_file)		
			id_ = image_file[0].split("/")[-1].split(".")[0]
			scipy.misc.imsave(sample_dir + str(id_)+'_'+str(i) + ".jpg", np.array(g_track_dict["fake_image"][0]))
if __name__ == '__main__':
	generate()
