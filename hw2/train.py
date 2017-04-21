import os
import sys
import time
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *
from structure import *
Embd_flag = true 

def train():
    #prepare data
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_feats, train_labels = [ getFeat(train_feat_dir + path) for path in train_data["feat_path"].values ], [ getLabel(train_label_dir + path) for path in train_data["label_path"].values ]
    test_labels = [ getLabel(test_label_dir + path) for path in test_data["label_path"].values ]
    all_merge_labels = [ label for labels in train_labels for label in labels ] + [ label for labels in test_labels for label in labels ]
    if Embd_flag is True:
		word_id, _, init_bias_vector,embd = buildEmbd(all_merge_labels)
	else:
		word_id, _, init_bias_vector = buildVocab(all_merge_labels)
    #initialize model
    model = VideoCaptionGenerator(
            video_size=video_size,
            video_step=video_step,
            caption_size=caption_size,
            caption_step=caption_step,
            hidden_size=hidden_size,
            batch_size=batch_size,
            init_bias_vector=init_bias_vector,
			pretrained_embd=embd
        )
    #build model
    tf_video_array, tf_video_array_mask, tf_caption_array, tf_caption_array_mask, tf_loss, tf_optimizer = model.buildModel()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=max_to_keep)
    #initialize variables
    tf.global_variables_initializer().run()
    #run epochs
    for it in range(num_iter):
        #shuffle index
        index_list_1 = []
        for _ in range(num_epoch_per_iter):
            index_list_1.extend(np.random.choice(len(train_labels), len(train_labels), replace=False))
        index_list_2 = []
        for labels in train_labels:
            temp_list = []
            for _ in range(num_epoch_per_iter // len(labels) + 1):
                temp_list.extend(np.random.choice(len(labels), len(labels), replace=False))
            temp_list = temp_list[:num_epoch_per_iter]
            index_list_2.append(temp_list)
        #batch
        start_time = time.time()
        for start, end in zip(range(0, len(index_list_1), batch_size), range(batch_size, len(index_list_1), batch_size)):
            #video, caption batch
            current_index_list_1 = index_list_1[start:end]
            current_video_batch = map(lambda x: train_feats[x], current_index_list_1)
            current_caption_batch = [ "<bos> " + train_labels[x][index_list_2[x].pop()] + " <eos>" for x in current_index_list_1 ]
            current_caption_id_batch = [ [ word_id[word] for word in sentence.lower().split(" ") ] for sentence in current_caption_batch ]
            #video_array
            video_array = np.zeros((batch_size, video_step, video_size), dtype="float32")
            for index, video in enumerate(current_video_batch):
                video_array[index] = video
            #video_array_mask
            video_array_mask = np.zeros((batch_size, video_step))
            #caption_array
            caption_array = np.zeros((batch_size, caption_step), dtype="int32")
            for index in range(len(current_caption_id_batch)):
                caption_array[index, :len(current_caption_id_batch[index])] = current_caption_id_batch[index]
            #caption_array_mask
            caption_array_mask = np.zeros((batch_size, caption_step))
            nonzero_length = np.array(map(lambda x: (x != 0).sum() - 1, caption_array))
            for index, row in enumerate(caption_array_mask):
                row[:nonzero_length[index]] = 1
            #loss
            fetch_dict = {
                "loss":tf_loss,
                "optimizer":tf_optimizer
            }
            feed_dict = {
                tf_video_array:video_array,
                tf_video_array_mask:video_array_mask,
                tf_caption_array:caption_array,
                tf_caption_array_mask:caption_array_mask
            }
            track_dict = session.run(fetch_dict, feed_dict)
            #print
            sys.stdout.write("\rBatchID: {0}, Loss: {1}".format(start / batch_size, track_dict["loss"]))
            sys.stdout.flush()
            if np.mod(start, batch_size * num_batch_per_epoch) == 0:
                end_time = time.time()
                sys.stdout.write("\nEpochID: {0}, Loss: {1}, Time: {2}\n".format(start / (batch_size * num_batch_per_epoch) + it * num_epoch_per_iter, track_dict["loss"], end_time - start_time))
                start_time = time.time()                
            if np.mod(start, batch_size * num_batch_per_epoch * save_num_epoch) == 0:
                print "Epoch ", start / (batch_size * num_batch_per_epoch) + it * num_epoch_per_iter, " is done. Saving the model..."
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                saver.save(session, model_dir, global_step=start/(batch_size*num_batch_per_epoch)+it*num_epoch_per_iter)      

if __name__ == "__main__":
    train()
