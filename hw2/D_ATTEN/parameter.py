#parameter
video_size = 4096
video_step = 80
caption_size = 2082
caption_step = 12
hidden_size = 256
batch_size = 10
output_keep_prob = 1
num_epoch = 3000
learning_rate = 0.001
save_per_epoch = 10
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
model_dir = "models/" + "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_epoch", str(num_epoch),
        "learning_rate", str(learning_rate)
    )) + "/"
test_model_path = model_dir + "-" + str(test_model_version)