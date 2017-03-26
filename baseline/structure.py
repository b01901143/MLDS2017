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

def run_epoch(epoch, session, input_figure, model_figure, is_training=False):
    total_cost_per_epoch = 0.0
    total_num_steps_per_epoch = 0
    initial_state = session.run(model_figure.initial_state)
    fetch_dict = {
        "logits": model_figure.logits,
        "cost": model_figure.cost,
        "x": input_figure.data,
        "y_": input_figure.labels,
        "cost_vector": model_figure.cost_vector
    }
    if is_training == True:
        fetch_dict["train_optimizer"] = model_figure.train_optimizer
    answers = []
    for batch in range(input_figure.num_batch):
        feed_dict = {}
        for i, (c, h) in enumerate(model_figure.initial_state):
            feed_dict[c] = initial_state[i].c
            feed_dict[h] = initial_state[i].h
        track_dict = session.run(fetch_dict, feed_dict)
        if input_figure.name == "TestInputFigure":
        	logits = track_dict["logits"]        	
	        cost_vector = []
	        answer_cost = 0.
	        for idx in range(5):
				cost_vector.append( np.sum(track_dict["cost_vector"][5*idx: 5*idx+5]) / input_figure.batch_size )
	        answers.append( str(chr(97 + np.argmin(cost_vector))) )
	        answer_cost += np.min(cost_vector)
        total_cost_per_epoch += track_dict["cost"]
        total_num_steps_per_epoch += input_figure.num_steps
        if is_training and batch % (input_figure.num_batch // 10) == 10:
            if input_figure.name == "TestInputFigure":
                print("%.3f perplexity: %.3f" % (batch * 1.0 / input_figure.num_batch, np.exp(answer_cost / total_num_steps_per_epoch)))
            else:
                print("%.3f perplexity: %.3f" % (batch * 1.0 / input_figure.num_batch, np.exp(total_cost_per_epoch / total_num_steps_per_epoch)))	
    if input_figure.name == "TestInputFigure":
        if not os.path.exists(answer_path):
            os.makedirs(answer_path)
        f_out = open(answer_path + str(epoch) + "_" + str(np.exp(total_cost_per_epoch / total_num_steps_per_epoch)) + ".csv", 'w') 
        f_out.write("id,answer\n")  
        for idx in range(len(answers)):
        	f_out.write(str(idx+1)+","+answers[idx]+"\n")	
        f_out.close()
    return np.exp(total_cost_per_epoch / total_num_steps_per_epoch)

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
