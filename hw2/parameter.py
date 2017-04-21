#parameter
vocab_size =30000
video_size = 4096
video_step = 80
caption_size = 6193 #fix
caption_step = 42
hidden_size = 300 # fix using pretrained embedding:300
batch_size = 10
num_iter = 10
num_epoch_per_iter = 37 #fix
num_batch_per_epoch = 145 #fix
save_num_epoch = 10
learning_rate = 0.001
test_model_version = 0

#for other settings
per_process_gpu_memory_fraction = 0.8
max_to_keep = 10

#path
train_feat_dir = "./data/training/feat/"
train_label_dir = "./data/training/label/"
train_info_path = "./data/training/info.csv"
test_feat_dir = "./data/testing/feat/"
test_label_dir = "./data/testing/label/"
test_info_path = "./data/testing/info.csv"
model_dir = "model/" + "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_iter", str(num_iter),
        "learning_rate", str(learning_rate)
    )) + "/"
test_model_path = model_dir + "-" + str(test_model_version)
