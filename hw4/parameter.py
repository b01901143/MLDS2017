#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 120 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000

#optimizer
learning_rate = 0.0002
beta1 = 0.5

#train
restore_flag = False
restore_version = 595
num_epoch = 600
save_num_batch = 50
save_num_epoch = 10
max_to_keep = 100

#path
'''
sample_training_info_path = "./info/sample_training_info"
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
sample_dir = "./samples/"
'''