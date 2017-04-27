#parameters
video_size = 4096
video_step = 80
caption_size = 4000 
#What does this mean??
#6193 for all embd labels, 6043 for non pretrained all labels
caption_step = 42
hidden_size = 300 #use 300 if using glove
batch_size = 100
output_keep_prob = 1
num_epoch = 2500
learning_rate = 0.001
save_per_epoch = 10
beam_width = 1
test_model_version = 0
#for other settings
per_process_gpu_memory_fraction = 0.8
max_to_keep = 10
Embd_flag = True

#paths
train_feat_dir, train_label_dir, train_info_path = "../data/training/feat/", "../data/training/label/", "../data/training/info.csv"
test_feat_dir, test_label_dir, test_info_path  = "../data/testing/feat/", "../data/testing/label/", "../data/testing/info.csv"
word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path =  "../dic/word_dic", "../dic/id_dic", "../dic/init_bias_dic", "../dic/embed_dic"
test_label_path = "../data/testing/label.json"
model_dir = "models/" + "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_epoch", str(num_epoch),
        "learning_rate", str(learning_rate)
    )) + "/"

