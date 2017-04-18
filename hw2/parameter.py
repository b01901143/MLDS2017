#parameter
video_size = 4096
video_step = 80
caption_size = 1932
caption_step = 12
hidden_size = 256
batch_size = 10
num_epoch = 500
save_per_epoch = 10
learning_rate = 0.001

#for other settings
per_process_gpu_memory_fraction = 0.8
max_to_keep = 10 #default 5 recent checkpoints

#path
train_feat_dir = "./data/training/feat/"
train_info_path = "./data/training/info.csv"
test_feat_dir = "./data/testing/feat/"
test_info_path = "./data/testing/info.csv"
model_path = "model/" + "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_epoch", str(num_epoch),
        "learning_rate", str(learning_rate)
    )) + "/"
