import os
import sys
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from utility import *
from parameter import *

class VideoCaptionGenerator():
    def __init__(self, video_size, video_step, caption_size, caption_step, hidden_size, batch_size):
        #parameters
        self.video_size = video_size
        self.video_step = video_step
        self.caption_size = caption_size
        self.caption_step = caption_step
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        #encode layers
        with tf.device("/cpu:0"):
            self.caption_encode_W = tf.Variable(tf.random_uniform([caption_size, hidden_size], -0.1, 0.1), name="caption_encode_W")
        self.video_encode_W = tf.Variable(tf.random_uniform([video_size, hidden_size], -0.1, 0.1), name="video_encode_W")
        self.video_encode_b = tf.Variable(tf.zeros([hidden_size]), name="video_encode_b")        
        #lstm layers
        self.lstm_1 = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)
        self.lstm_2 = tf.contrib.rnn.BasicLSTMCell(hidden_size, state_is_tuple=False)
        #decode layers
        self.caption_decode_W = tf.Variable(tf.random_uniform([hidden_size, caption_size], -0.1, 0.1), name="caption_decode_W")
        self.caption_decode_b = tf.Variable(tf.zeros([caption_size]), name="caption_decode_b")
    def buildModel(self):
        #placeholders
        tf_video_array = tf.placeholder(tf.float32, [self.batch_size, self.video_step, self.video_size])
        tf_video_array_mask = tf.placeholder(tf.float32, [self.batch_size, self.video_step])
        tf_caption_array = tf.placeholder(tf.int32, [self.batch_size, self.caption_step])
        tf_caption_array_mask = tf.placeholder(tf.float32, [self.batch_size, self.caption_step])
        #tensors
        video_array_flat = tf.reshape(tf_video_array, [-1, self.video_size])
        video_embed = tf.nn.xw_plus_b(video_array_flat, self.video_encode_W, self.video_encode_b)
        video_embed = tf.reshape(video_embed, [self.batch_size, self.video_step, self.hidden_size])
        state_1 = tf.zeros([self.batch_size, self.lstm_1.state_size])
        state_2 = tf.zeros([self.batch_size, self.lstm_2.state_size])
        padding = tf.zeros([self.batch_size, self.hidden_size])
        tf_loss = 0.0
        with tf.varaible_scope("RNN"):
            #encoding stage
            for step in range(self.video_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1(video_embed[:, i, :], state_1)
                with tf.variable_scope("LSTM2"):
                    output_2, state_2 = self.lstm_2(tf.concat([padding, output_1], 1), state_2)
            #decoding stage
            for i in range(self.caption_step-1):
                with tf.device("/cpu:0"):
                    caption_embed = tf.nn.embedding_lookup(self.caption_encode_W, tf.caption_array[:, i])
                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1(padding, state_1)
                with tf.variable_scope("LSTM2"):
                    output_2, state_2 = self.lstm_2(tf.concat([caption_embed, output_1], 1), state_2)
                #cross entropy
                logits = tf.nn.xw_plus_b(output2, self.caption_decode_W, self.caption_decode_b)
                labels = tf.sparse_to_dense(
                    tf.concat([tf.expand_dims(tf.range(self.batch_size), 1), tf.expand_dims(tf_caption_array[:, i+1], 1)], 1),
                    tf.stack([self.batch_size, self.caption_size]),
                    1.0,
                    0.0
                )
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels) * caption_array_mask[:, i]
                tf_loss += tf.reduce_sum(cross_entropy) / self.batch_size
        tf_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(tf_loss)     
        return tf_video_array, tf_video_array_mask, tf_caption_array, tf_caption_array_mask, tf_loss, tf_optimizer

    def buildGenerator(self):
        
        # tf.get_variable_scope().reuse_variables()
        
        video = tf.placeholder(tf.float32, [1, self.video_step, self.video_size])

        video_flat = tf.reshape(video, [-1, self.video_size])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.video_step, self.hidden_size])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.hidden_size])

        generated_words = []

        probs = []
        embeds = []

        with tf.variable_scope("RNN"):

            for i in range(0, self.video_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(image_emb[:, i, :], state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)

            for i in range(0, self.caption_step):
                tf.get_variable_scope().reuse_variables()

                if i == 0:
                    with tf.device('/cpu:0'):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, tf.ones([1], dtype=tf.int64))

                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)

                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)

                logit_words = tf.nn.xw_plus_b( output2, self.embed_word_W, self.embed_word_b)
                max_prob_index = tf.argmax(logit_words, 1)[0]
                generated_words.append(max_prob_index)
                probs.append(logit_words)

                with tf.device("/cpu:0"):
                    current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                    current_embed = tf.expand_dims(current_embed, 0)

                embeds.append(current_embed)

        return video, video_mask, generated_words, probs, embeds

def train():
    #prepare data
    train_data, all_data = getInfo(train_info_path), pd.concat([getInfo(train_info_path), getInfo(test_info_path)])
    word_id, id_word = buildVocab(all_data["label_sentence"].values)
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
    tf_video_array, tf_video_array_mask, tf_caption_array, tf_caption_array_mask, tf_loss, tf_optimizer = model.buildModel()
    #build session, saver
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=per_process_gpu_memory_fraction)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
    saver = tf.train.Saver(max_to_keep=100)
    #initialize variables
    tf.global_variables_initializer().run()
    #run epochs
    for epoch in range(num_epoch):
        #shuffle
        index_list = np.arange(len(train_data))
        np.random.shuffle(index_list)
        current_train_data = train_data.ix[index_list]
        #batch
        start_time = time.time()
        for start, end in zip(range(0, len(current_train_data), batch_size), range(batch_size, len(current_train_data), batch_size)):
            #video, caption batch
            current_batch = current_train_data[start:end]
            current_video_batch = map(lambda x: np.load(train_feat_dir + x), current_batch["feat_path"].values)
            current_caption_batch = [ "<bos> " + sentence + " <eos>" for sentence in current_batch["label_sentence"].values ]
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
                "loss":tf_loss
                "optimizer":tf_optimizer
            }
            feed_dict = {
                tf_video_array:video_array
                tf_video_array_mask:video_array_mask
                tf_caption_array:caption_array
                tf_caption_array_mask:caption_array_mask
            }
            session.run(fetch_dict, feed_dict)
            #print
            sys.stdout.write("\rBatchID: {0}, Loss: {1}".format(start / batch_size, loss_eval))
            sys.stdout.flush()
        end_time = time.time()
        sys.stdout.write("Epoch: {0}, Loss: {1}, Time: {2}".format(epoch, loss_eval, start_time - end_time))
        #save
        if np.mod(epoch, save_per_epoch) == 0:
            print "Epoch ", epoch, " is done. Saving the model..."
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(session, model_path, global_step=epoch)            

if __name__ == "__main__":
    train()
