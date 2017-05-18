#model
image_size = 64
caption_size = 4800
embedding_size = 256
noise_size = 100
g_channel_size = 64
d_channel_size = 64
batch_size = 64

#optimizer
learning_rate = 0.0002
beta1 = 0.5

#train
restore_flag = False
restore_version = 0
num_epoch = 600
save_num_batch = 50

#path
training_sample_text_path = "./info/sample_training_text.txt"
training_text_image_path = "./info/training_text_image"
testing_sample_text_path = "./info/sample_testing_text.txt"
testing_text_image_path = "./info/testing_text_image"
model_dir = "./models/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
save_dir = "./saves/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
sample_dir = "./samples/" + "_".join((
		"image_size", str(image_size),
		"caption_size", str(caption_size),
		"embedding_size", str(embedding_size),
		"noise_size", str(noise_size),
		"g_channel_size", str(g_channel_size),
		"d_channel_size", str(d_channel_size),
		"batch_size", str(batch_size),
		"learning_rate", str(learning_rate),
		"beta1", str(beta1)
    )) + "/"
training_save_caption_path = save_dir + "caption.txt" 
testing_save_caption_path = sample_dir + "caption.txt"
