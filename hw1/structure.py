import os
import numpy as np
import tensorflow as tf

from parameter import *
from utility import *

def embedding_layer(x):
    embedding_weights = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary, num_units], name="embedding_weights")
    embed = tf.nn.embedding_lookup(embedding_weights, x)
    return embed

def lstm_cell():
    return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=forget_bias, state_is_tuple=True)

def cell_wrapper():
    return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell(), input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)

def softmax_layer(x):
    softmax_weights = tf.get_variable(dtype=tf.float32, shape=[num_units, num_vocabulary], name="softmax_weights")
    softmax_biases = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary], name="softmax_biases")
    product = tf.add(tf.matmul(x, softmax_weights), softmax_biases)
    return product

class InputFigure(object):
    def __init__(self, data, batch_size, num_steps, name):
        self.name = name
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_batch = ((len(data) // batch_size) - 1) // num_steps
        if name == "TestInputFigure":
            self.data, self.labels, self.data_r = ptb_producer_test(data, batch_size, num_steps, name)
            self.num_batch = test_num_batch
        else:
            self.data, self.labels, self.data_r = ptb_producer(data, batch_size, num_steps, name)

class ModelFigure(object):
    def __init__(self, input_figure, is_training=False):
        self.input_figure = input_figure
        with tf.device("/cpu:0"):
            self.input_layer = embedding_layer(self.input_figure.data)
        if is_training:
            self.input_layer = tf.nn.dropout(self.input_layer, output_keep_prob)
        if is_training:
            multilayers = tf.contrib.rnn.MultiRNNCell(cells=[ lstm_cell() for _ in range(num_layers) ], state_is_tuple=True)
        else:
            multilayers = tf.contrib.rnn.MultiRNNCell(cells=[ cell_wrapper() for _ in range(num_layers) ], state_is_tuple=True)
        self.initial_state = multilayers.zero_state(input_figure.batch_size, tf.float32)
        current_outputs = []
        current_state = self.initial_state
        with tf.variable_scope("RNN"):
            for step in range(input_figure.num_steps):
                if step > 0:
                    tf.get_variable_scope().reuse_variables()
                current_output, current_state = multilayers(self.input_layer[:, step, :], current_state)
                current_outputs.append(current_output)
        self.final_output = tf.reshape(tf.concat(axis=1, values=current_outputs), [-1, num_units])
        self.final_state = current_state
        self.logits = softmax_layer(self.final_output)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [self.logits],
            [tf.reshape(input_figure.labels, [-1])],
            [tf.ones([input_figure.batch_size * input_figure.num_steps], dtype=tf.float32)]
        )
        self.cost = tf.reduce_sum(loss) / input_figure.batch_size
        self.cost_vector = loss
        if is_training == False:
            return
        self.lr = tf.Variable(0.0, trainable=False)
        trainable_variables = tf.trainable_variables()
        gradients, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainable_variables), max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.lr)
        self.train_optimizer = optimizer.apply_gradients(
            zip(gradients, trainable_variables),
            global_step=tf.contrib.framework.get_or_create_global_step()
        )
        self.new_lr = tf.placeholder(dtype=tf.float32, shape=[], name="new_learning_rate")
        self.assign_new_lr = tf.assign(self.lr, self.new_lr)
