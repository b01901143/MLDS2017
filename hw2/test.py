import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *

def test():
    #prepare data
    test_data, all_data = getInfo(test_info_path), pd.concat([getInfo(train_info_path), getInfo(test_info_path)])
    _, id_word = buildVocab(all_data["label_sentence"].values)
    #initialize model
    model = VideoCaptionGenerator(
            video_size=video_size,
            video_step=video_step,
            caption_size=caption_size,
            caption_step=caption_step,
            hidden_size=hidden_size,
            batch_size=batch_size
        )
    #build model
    tf_video_array, tf_video_array_mask, tf_caption_array_id = model.buildGenerator()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=100)
    #restore variables
    saver.restore(session, test_model_path)
    #run testing
    for index, feat_path in enumerate(test_data["feat_path"]):
        print "VideoID: " + index + "Path: " + feat_path
        video_array = np.load(test_feat_dir + feat_path) 
        video_array_mask = np.ones(self.video_step, self.video_size)
        #caption_array
        fetch_dict = {
            "caption_array_id":tf_caption_array_id
        }
        feed_dict = {
            "video_array":tf_video_array
            "video_array_mask":tf_video_array_mask
        }
        track_dict = session.run(fetch_dict, feed_dict)
        caption_array_id = track_dict["caption_array_id"]
        caption_array = id_word[caption_array_id]
        print caption_array    

if __name__ == "__main__"
    test()
