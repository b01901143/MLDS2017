import json
import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *

def test():
    #prepare data
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_label = [ json.load(open(train_label_dir + path)) for path in train_data["label_path"].values ]
    test_label = [ json.load(open(test_label_dir + path)) for path in test_data["label_path"].values ]
    _, id_word, init_bias_vector = buildVocab(train_label + test_label)
    #initialize model
    model = VideoCaptionGenerator(
            video_size=video_size,
            video_step=video_step,
            caption_size=caption_size,
            caption_step=caption_step,
            hidden_size=hidden_size,
            batch_size=batch_size,
            output_keep_prob=output_keep_prob,
            init_bias_vector=init_bias_vector
        )
    #build model
    tf_video_array, tf_video_array_mask, tf_caption_array_id = model.buildGenerator()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    #restore variables
    saver.restore(session, test_model_path)
    #run testing
    for index, feat_path in enumerate(test_data["feat_path"]):
        print "VideoID: " + str(index) + " Path: " + feat_path
        video_array = np.load(test_feat_dir + feat_path)[None,...] 
        video_array_mask = np.ones((video_array.shape[0], video_array.shape[1]))
        #caption_array
        fetch_dict = {
            "caption_array_id":tf_caption_array_id
        }
        feed_dict = {
            tf_video_array:video_array,
            tf_video_array_mask:video_array_mask
        }
        track_dict = session.run(fetch_dict, feed_dict)
        caption_array_id = track_dict["caption_array_id"]
        caption_array = [ id_word[idx] for arr in caption_array_id for idx in arr ]
        print caption_array   

if __name__ == "__main__":
    test()