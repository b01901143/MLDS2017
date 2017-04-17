#paremeters
video_size = 4096
video_step = 80
caption_size = 1932
caption_step = 20
hidden_size = 256
num_epoch = 500
batch_size = 10
learning_rate = 0.001

#path
model_path = "_".join((
        "num_vocabulary", str(caption_size),
        "hidden_size", str(hidden_size),
        "num_epoch", str(num_epoch),
        "learning_rate", str(learning_rate)
    )) + "/"
