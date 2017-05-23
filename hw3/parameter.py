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
restore_flag = True
restore_version = 599
num_epoch = 600
save_num_batch = 50

#path
sample_training_text_path = "./info/sample_training_text.txt"
sample_training_info_path = "./info/sample_training_info"
sample_testing_text_path = "./info/sample_testing_text.txt"
sample_testing_info_path = "./info/sample_testing_info"
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
result_training_dir = "./results/training/" + "_".join((
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
result_testing_dir = "./results/testing/" + "_".join((
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
result_training_caption_path = result_training_dir + "caption.txt" 
result_testing_caption_path = result_testing_dir + "caption.txt"
