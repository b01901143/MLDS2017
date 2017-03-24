import reader
import numpy as np
import tensorflow as tf
import sys
import divide
# import word2vec as w2v
from parse_question import *
from preprocessing import *



#vocabulary
num_vocabulary = reader.num_vocabulary
#defined in reader!!!

#initializer
initial_scale = 0.05

#embedding layer
# pretrainEmbd=w2v.embd_table()
pretrained=None

#dropout_layer
input_keep_prob = 1.0
output_keep_prob = 0.65

#hidden_layer
num_units = 200
forget_bias = 0.0
num_layers = 1

#learning_rate
learning_rate = 0.5
learning_rate_decay = 0.5
learning_rate_decay_param = 1
max_grad_norm = 5

#batch, epoch
num_epoch = 5
train_batch_size = valid_batch_size = 20
train_num_steps = valid_num_steps = 20
test_batch_size = test_num_steps = 5


#path
data_path = "./data/sets/cut/"
save_path = "./save/"

#data
# raw_data = reader.ptb_raw_data(data_path,pretrained,pretrainEmbd._word2id)
raw_data = reader.ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, _ = raw_data

if sys.argv[1] == "--reparse":
	print "re-generating training data ..."
	divide.reparse(train_num_steps)

#testing data


test_datasets, test_optionsets = get_questions(), get_options()
testing_data_batches, testing_num_batch = generate_testing_batches(test_datasets, test_optionsets)

for i in range(len(testing_data_batches)):
	for j in range(len(testing_data_batches[i])):
		testing_data_batches[i,j] = [word_to_id[word] if word in word_to_id else num_vocabulary-1 for word in testing_data_batches[i,j] ]
# zero_padding = np.zeros((test_num_steps, 6),dtype=np.int)
# test_data =  np.hstack(  ( zero_padding, np.reshape( np.swapaxes(testing_data_batches, 0, 1), [test_num_steps, -1] ) )  )
test_data = np.reshape( np.swapaxes(testing_data_batches, 0, 1), -1 )



def embedding_layer(x):
# if pretrained x are id, otherwise words
    if pretrained==None:
        embedding_weights = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary, num_units], name="embedding_weights")
        embed = tf.nn.embedding_lookup(embedding_weights, x)
    else:
		embed = tf.nn.embedding_lookup(pretrainEmbd.lookupEmbd(),x)
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

        # print track_dict["x"]
        # raw_input()
        # print track_dict["y_"]
        # raw_input()

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

        f_out = open("_ans_epoch_" + str(epoch) + "_" + str(np.exp(total_cost_per_epoch / total_num_steps_per_epoch)) + ".csv", 'w')
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
        	self.data, self.labels, self.data_r = reader.ptb_producer_test(data, batch_size, num_steps, name)
        	self.num_batch = 1040
        else:
        	self.data, self.labels, self.data_r = reader.ptb_producer(data, batch_size, num_steps, name)

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

with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-initial_scale, initial_scale)
    with tf.name_scope("Train"):
        print("build train model")
        train_input_figure = InputFigure(train_data, train_batch_size, train_num_steps, "TrainInputFigure")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model_figure = ModelFigure(input_figure=train_input_figure, is_training=True)
        tf.summary.scalar("Training Loss", train_model_figure.cost)
        tf.summary.scalar("Learning Rate", train_model_figure.lr)
    with tf.name_scope("Valid"):
		print("build valid model")
		valid_input_figure = InputFigure(valid_data, valid_batch_size, valid_num_steps, "ValidInputFigure")
		with tf.variable_scope("Model", reuse=True, initializer=initializer):
			valid_model_figure = ModelFigure(input_figure=valid_input_figure, is_training=False)
		tf.summary.scalar("Validation Loss", valid_model_figure.cost)
    with tf.name_scope("Test"):
		print("build test model")
		test_input_figure = InputFigure(test_data, test_batch_size, test_num_steps, "TestInputFigure")
		with tf.variable_scope("Model", reuse=True, initializer=initializer):
			test_model_figure = ModelFigure(input_figure=test_input_figure, is_training=False)
    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as session:
    	# test_perplexity = run_epoch(session, test_input_figure, test_model_figure)
     #    print("Test Perplexity: %.3f" % test_perplexity)
        for i in range(num_epoch):
            session.run(train_model_figure.assign_new_lr, feed_dict={train_model_figure.new_lr: learning_rate * (learning_rate_decay ** max(i + 1 - learning_rate_decay_param, 0.0))})
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model_figure.lr)))
            train_perplexity = run_epoch(i,session, train_input_figure, train_model_figure, is_training=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(i,session, valid_input_figure, valid_model_figure)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            test_perplexity = run_epoch(i,session, test_input_figure, test_model_figure)
            print("Test Perplexity: %.3f" % test_perplexity)
        if save_path:
            print("Saving model to %s." % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)
