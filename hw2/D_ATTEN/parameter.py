#parameter
video_size = 4096
video_step = 80
caption_size = 6193 # 6193 for all embd labels , 6043 for non pretrained all labels
caption_step = 22
hidden_size = 300 # use 300 if using glove
batch_size = 128
output_keep_prob = 1
num_epoch = 2500
learning_rate = 0.001
save_per_epoch = 10
beam_width = 1
#for other settings
per_process_gpu_memory_fraction = 0.8
max_to_keep = 10
Embd_flag = True

#path
train_feat_dir = "../data/training/feat/"
train_label_dir = "../data/training/label/"
train_info_path = "../data/training/info.csv"
test_feat_dir = "../data/testing/feat/"
test_label_dir = "../data/testing/label/"
test_label_all = "../data/testing/label.json"
test_info_path = "../data/testing/info.csv"
model_dir = "models/" + "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_epoch", str(num_epoch),
        "learning_rate", str(learning_rate)
    )) + "/"

