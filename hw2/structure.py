import tensorflow as tf
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
                #loss
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
        #placeholders
        tf_video_array = tf.placeholder(tf.float32, [1, self.video_step, self.video_size])
        tf_video_array_mask = tf.placeholder(tf.float32, [1, self.video_step])
        #tensors
        video_array_flat = tf.reshape(tf_video_array, [-1, self.video_size])
        video_embed = tf.nn.xw_plus_b(video_array_flat, self.video_encode_W, self.video_encode_b)
        video_embed = tf.reshape(video_embed, [1, self.video_step, self.hidden_size])
        state_1 = tf.zeros([1, self.lstm_1.state_size])
        state_2 = tf.zeros([1, self.lstm_2.state_size])
        padding = tf.zeros([1, self.hidden_size])=
        tf_caption_array_id = []
        with tf.variable_scope("RNN"):
            #encoding stage
            for i in range(self.video_step):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1(video_embed[:, i, :], state_1)
                with tf.variable_scope("LSTM2"):
                    output_2, state_2 = self.lstm_2(tf.concat([padding, output_1], 1), state_2)
            #decoding stage
            for i in range(self.caption_step-1):
                with tf.device("./cpu:0"):
                    if i == 0:
                        caption_embed = tf.nn.embedding_lookup(self.caption_encode_W, tf.ones([1], dtype=tf.int64))
                    else:
                        caption_embed = tf.nn.embedding_lookup(self.caption_encode_W, max_prob_index)
                        caption_embed = tf.expand_dims(caption_embed, 0)                        
                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output_1, state_1 = self.lstm_1(padding, state_1)
                with tf.variable_scope("LSTM2"):
                    output_2, state_2 = self.lstm_2(tf.concat([caption_embed, output_1], 1), state_2)
                #caption_array_id
                logits = tf.nn.xw_plus_b(output_2, self.caption_decode_W, self.caption_decode_b)
                max_prob_index = tf.argmax(logits, 1)[0]
                tf_caption_array_id.append(max_prob_index)
        return tf_video_array, tf_video_array_mask, tf_caption_array_id
