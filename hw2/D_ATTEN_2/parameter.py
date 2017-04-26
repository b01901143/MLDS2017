#parameter
video_size = 4096
caption_size = 4000
video_step = 80
caption_step = 42
hidden_size = 256
batch_size = 32
num_layer = 1
num_sampled = 512
learning_rate = 0.2
learning_rate_decay_factor = 0.99
max_gradient_norm = 5.0
num_epoch = 500

#for other settings
max_train_data_size = 0
steps_per_check = 1
steps_per_save = 20

#path
data_dir = "./data/"
model_dir = "./models/"
train_feat_dir = "./data/training/feat/"
train_label_dir = "./data/training/label/"
train_info_path = "./data/training/info.csv"
test_feat_dir = "./data/testing/feat/"
test_label_dir = "./data/testing/label/"
test_info_path = "./data/testing/info.csv"
