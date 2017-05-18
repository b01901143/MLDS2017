import os
from utility import *
from parameter import *
from model import *

def generate():
	#get text_image info
	testing_sample_text_dict = readSampleText(testing_sample_text_path)
	testing_text_image_list = readTextToImage(testing_text_image_path)
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
	for batch in range(len(testing_text_image_list) // 1):
		current_infer = testing_text_image_list[batch : (batch+1) * 1]
		caption, noise, image_file = getInferData(current_infer)
		feed_dict = {
			input_tensor['caption']: caption,
			input_tensor['noise']: noise
		}
		g_fetch_dict = {
			"fake_image": output_tensor["fake_image"]
		}
		g_track_dict = session.run(g_fetch_dict, feed_dict=feed_dict)
		if not os.path.exists(sample_dir):
			os.makedirs(sample_dir)
		sys.stdout.write("\rBatchID: {0}, Saving Image...".format(batch))
		sys.stdout.flush()
		sampleImage(sample_dir, g_track_dict["fake_image"], batch)		

if __name__ == '__main__':
	generate()
