#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 512 # embedding dimension
HIDDEN_DIM = 512 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 1
PRE_EPOCH_NUM = 20 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 2*EMB_DIM
dis_filter_sizes = [1, 2, 3, 4, 5, 6]
dis_num_filters = [100, 200, 200, 200, 200, 100]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = BATCH_SIZE

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_EPOCH = 301
positive_file = 'data/real_data.txt'
negative_file = 'data/generator_sample.txt'
eval_file = 'data/eval_file.txt'
generated_num = BATCH_SIZE*3 #10000
vocab_size = 20000
#train
restore_flag = False
num_epoch = 600
save_num_batch = 50
max_to_keep = 100

# path
model_dir = './data/model/'
sample_dir= './data/sample/'
