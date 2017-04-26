import os
import sys
import time
import math
import json
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
    word_id, id_word, _ = buildVocab(train_label + test_label)
    #initialize model
    model = VideoCaptionGenerator(
        video_size=video_size,
        video_step=video_step,
        caption_size=caption_size,
        caption_step=caption_step,
        hidden_size=hidden_size,
        batch_size=batch_size,
        num_layer=num_layer,
        num_sampled=num_sampled,
        learning_rate=learning_rate,
        learning_rate_decay_factor=learning_rate_decay_factor,
        max_gradient_norm=max_gradient_norm,
    )
    model.batch_size = 1
    #build model
    tf_video_array, tf_caption_array, tf_caption_array_mask, tf_outputs = model.buildGenerator()
    #build session, saver
    session = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print "restore model from %s..." % ckpt.model_checkpoint_path
        saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print "create a new model..."
        session.run(tf.global_variables_initializer())
    #run testing
    for index, (feat_path, label_path) in enumerate(zip(test_data["feat_path"], test_data["label_path"])): 
        print("VideoID: " + str(index) + " Path: " + feat_path)        
        current_caption_batch = [ "<bos> " + json.load(open(test_label_dir + label_path)) + " <eos>" ]
        current_caption_id_batch = [ [ word_id[word] for word in sentence.lower().split(" ") if word in word_id ] for sentence in current_caption_batch ]        
        #video_array(a list of 2D tensors)
        temp_video_array = np.array(np.load(test_feat_dir + feat_path)[None,...])
        temp_video_array = np.transpose(np.array(temp_video_array), (1, 0, 2))
        video_array = [ arr for arr in temp_video_array ] 
        #caption_array(a list of 1D tensors)
        temp_caption_array = np.zeros((model.batch_size, caption_step), dtype=np.int32)
        for index in range(len(current_caption_id_batch)):
           temp_caption_array[index, :len(current_caption_id_batch[index])] = current_caption_id_batch[index]
        #caption_array_mask(a list of 1D tensors)
        temp_caption_array_mask = np.zeros((model.batch_size, caption_step))
        nonzero_length = np.array(map(lambda x: (x != 0).sum() - 1, temp_caption_array))
        for index, row in enumerate(temp_caption_array_mask):
            row[:nonzero_length[index]] = 1            
        temp_caption_array = np.transpose(temp_caption_array, (1, 0))
        caption_array = [ arr for arr in temp_caption_array ]
        temp_caption_array_mask = np.transpose(temp_caption_array_mask, (1, 0))
        caption_array_mask = [ arr for arr in temp_caption_array_mask ]
        fetch_dict = {
            "outputs":tf_outputs,
        }
        feed_dict = {}
        for t, l in zip(tf_video_array, video_array):
            feed_dict[t] = l
        for t, l in zip(tf_caption_array, caption_array):
            feed_dict[t] = l
        for t, l in zip(tf_caption_array_mask, caption_array_mask):
            feed_dict[t] = l
        track_dict = session.run(fetch_dict, feed_dict) 
        outputs = [ int(np.argmax(logit)) for logit in track_dict["outputs"]  ]  
        outputs = [ output for output in outputs if output != 2 ]
        outputs = [ id_word[output] for output in outputs ]
        print outputs
        sys.stdout.flush()
        
if __name__ == "__main__":
    test()
