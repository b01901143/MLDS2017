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

def train():
    #prepare data
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_label = [ json.load(open(train_label_dir + path)) for path in train_data["label_path"].values ]
    test_label = [ json.load(open(test_label_dir + path)) for path in test_data["label_path"].values ]  
    word_id, _, _ = buildVocab(train_label + test_label)
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
    #build model
    tf_video_array, tf_caption_array, tf_caption_array_mask, tf_losses, tf_max_norm, tf_updates = model.buildModel()
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
    #run epochs
    current_step = 0
    step_time, step_loss = 0.0, 0.0
    previous_step_losses = []
    for epoch in range(num_epoch):
        #shuffle
        index_list = np.arange(len(train_data))
        np.random.shuffle(index_list)
        current_train_data = train_data.ix[index_list]
        #batch
        for start, end in zip(range(0, len(current_train_data), batch_size), range(batch_size, len(current_train_data), batch_size)):      
            #video, caption batch
            start_time = time.time()
            current_batch = current_train_data[start:end]
            current_video_batch = map(lambda x: np.load(train_feat_dir + x), current_batch["feat_path"].values)
            current_caption_batch = [ "<bos> " + json.load(open(train_label_dir + path)) + " <eos>" for path in current_batch["label_path"].values ]
            current_caption_id_batch = [ [ word_id[word] for word in sentence.lower().split(" ") if word in word_id ] for sentence in current_caption_batch ]
            #video_array(a list of 2D tensors)
            temp_video_array = np.transpose(np.array(current_video_batch), (1, 0, 2))
            video_array = [ arr for arr in temp_video_array ]
            #caption_array(a list of 1D tensors)
            temp_caption_array = np.zeros((batch_size, caption_step), dtype=np.int32)
            for index in range(len(current_caption_id_batch)):
               temp_caption_array[index, :len(current_caption_id_batch[index])] = current_caption_id_batch[index]
            #caption_array_mask(a list of 1D tensors)
            temp_caption_array_mask = np.zeros((batch_size, caption_step))
            nonzero_length = np.array(map(lambda x: (x != 0).sum() - 1, temp_caption_array))
            for index, row in enumerate(temp_caption_array_mask):
                row[:nonzero_length[index]] = 1            
            temp_caption_array = np.transpose(temp_caption_array, (1, 0))
            caption_array = [ arr for arr in temp_caption_array ]
            temp_caption_array_mask = np.transpose(temp_caption_array_mask, (1, 0))
            caption_array_mask = [ arr for arr in temp_caption_array_mask ]       
            #loss
            fetch_dict = {
                "losses":tf_losses,
                "max_norm":tf_max_norm,
                "updates":tf_updates
            }
            feed_dict = {}
            for t, l in zip(tf_video_array, video_array):
               feed_dict[t] = l
            for t, l in zip(tf_caption_array, caption_array):
               feed_dict[t] = l
            for t, l in zip(tf_caption_array_mask, caption_array_mask):
               feed_dict[t] = l
            track_dict = session.run(fetch_dict, feed_dict)
            end_time = time.time()
            step_time += (end_time - start_time) / steps_per_check
            step_loss += track_dict["losses"] / steps_per_check
            current_step += 1
            if current_step % steps_per_check == 0:
                perplexity = math.exp(float(step_loss)) if step_loss < 300 else float("inf")
                record_global_step, record_learning_rate = session.run(model.global_step), session.run(model.learning_rate) 
                print "global step %d learning rate %.4f step-time %.2f perplexity %.2f" % (record_global_step, record_learning_rate, step_time, perplexity)
                if len(previous_step_losses) > 2 and step_loss > max(previous_step_losses[-3:]):
                    session.run(model.learning_rate_decay_op)
                previous_step_losses.append(step_loss)
                checkpoint_path = os.path.join(model_dir, "VideoCaptionGenerator.ckpt")
                step_time, step_loss = 0.0, 0.0
                sys.stdout.flush()
            if current_step % steps_per_save == 0:
                print "save model %d" 
                saver.save(session, checkpoint_path, global_step=model.global_step)

if __name__ == "__main__":
    train()
