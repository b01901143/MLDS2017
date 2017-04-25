import numpy as np
import tensorflow as tf
from function import *

class VideoCaptionGenerator(object):
    def __init__(
          self,
          video_size,
          video_step,
          caption_size,
          caption_step,
          hidden_size,
          batch_size,
          num_layer,
          num_sampled,
          learning_rate,
          learning_rate_decay_factor,
          max_gradient_norm,
    ):
        #parameters
        self.video_size = video_size
        self.video_step = video_step
        self.caption_size = caption_size
        self.caption_step = caption_step
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_layer = num_layer
        self.num_sampled = num_sampled
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=tf.float32)
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_decay_factor = learning_rate_decay_factor
        self.max_gradient_norm = max_gradient_norm
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * self.learning_rate_decay_factor)        
        #lstm_layers
        def single_cell():
            return tf.contrib.rnn.GRUCell(self.hidden_size)
        self.cell = single_cell()
        if num_layer > 1:
            self.cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(self.num_layer)])
        #output_layers
        self.output_projection_W_t = tf.get_variable("proj_w", [ self.caption_size, self.hidden_size ], dtype=tf.float32)
        self.output_projection_W = tf.transpose(self.output_projection_W_t)
        self.output_projection_b = tf.get_variable("proj_b", [ self.caption_size ], dtype=tf.float32)
        self.output_projection = (self.output_projection_W, self.output_projection_b)
        #loss_functions
        def sampled_loss(labels, logits):
            return tf.cast(
                tf.nn.sampled_softmax_loss(
                    weights=tf.cast(self.output_projection_W_t, tf.float32),
                    biases=tf.cast(self.output_projection_b, tf.float32),
                    labels=tf.reshape(labels, [-1, 1]),
                    inputs=tf.cast(logits, tf.float32),
                    num_sampled=self.num_sampled,
                    num_classes=self.caption_size
                ), tf.float32
            )
        self.softmax_loss_function = sampled_loss        
    def buildModel(self):
        #return tensors
        tf_video_array = [ tf.placeholder(tf.float32, shape=[ None, self.video_size ], name="tf_video_array{0}".format(step)) for step in range(self.video_step) ]
        tf_caption_array = [ tf.placeholder(tf.int32, shape=[ None ], name="tf_caption_array{0}".format(step)) for step in range(self.caption_step) ]
        tf_caption_array_mask = [ tf.placeholder(tf.float32, shape=[ None ], name="tf_caption_array_mask{0}".format(step)) for step in range(self.caption_step) ]
        with tf.name_scope(None, "top_wrapper", tf_video_array+tf_caption_array+tf_caption_array_mask):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=None):
                tf_outputs, _ = top_wrapper(
                    encoder_inputs=tf_video_array,
                    decoder_inputs=tf_caption_array[:self.caption_step-1],
                    cell=self.cell,
                    source_size=self.video_size,
                    target_size=self.caption_size, 
                    embedding_size=self.hidden_size,
                    output_projection=self.output_projection,
                    feed_previous=False,
                    dtype=tf.float32
                )
                tf_losses = tf.contrib.legacy_seq2seq.sequence_loss(
                    logits=tf_outputs,
                    targets=tf_caption_array[1:],
                    weights=tf_caption_array_mask[:self.caption_step-1],
                    softmax_loss_function=self.softmax_loss_function
                )      
        parameters = tf.trainable_variables()
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        max_gradients, tf_max_norm = tf.clip_by_global_norm(tf.gradients(tf_losses, parameters), self.max_gradient_norm)
        tf_updates = optimizer.apply_gradients(zip(max_gradients, parameters), global_step=self.global_step)
        return tf_video_array, tf_caption_array, tf_caption_array_mask, tf_losses, tf_max_norm, tf_updates
    def buildGenerator(self):
        #return tensors
        tf_video_array = [ tf.placeholder(tf.float32, shape=[ None, self.video_size ], name="tf_video_array{0}".format(step)) for step in range(self.video_step) ]
        tf_caption_array = [ tf.placeholder(tf.int32, shape=[ None ], name="tf_caption_array{0}".format(step)) for step in range(self.caption_step) ]
        tf_caption_array_mask = [ tf.placeholder(tf.float32, shape=[ None ], name="tf_caption_array_mask{0}".format(step)) for step in range(self.caption_step) ]    
        with tf.name_scope(None, "top_wrapper", tf_video_array+tf_caption_array+tf_caption_array_mask):
            with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=None):
                tf_outputs, _ = top_wrapper(
                    encoder_inputs=tf_video_array,
                    decoder_inputs=tf_caption_array[:self.caption_step-1],
                    cell=self.cell,
                    source_size=self.video_size,
                    target_size=self.caption_size,
                    embedding_size=self.hidden_size,
                    output_projection=self.output_projection,
                    feed_previous=True,   
                    dtype=tf.float32
                )
                tf_losses = tf.contrib.legacy_seq2seq.sequence_loss(
                    logits=tf_outputs,
                    targets=tf_caption_array[1:],
                    weights=tf_caption_array_mask[:self.caption_step-1],
                    softmax_loss_function=self.softmax_loss_function
                )
        if self.output_projection is not None:
            tf_outputs = tf.matmul(tf.reshape(tf_outputs, [-1, self.hidden_size]), self.output_projection[0]) + self.output_projection[1]      
        return tf_video_array, tf_caption_array, tf_caption_array_mask, tf_outputs
