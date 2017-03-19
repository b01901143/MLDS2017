#packages
import sys
import numpy as np
import tensorflow as tf
from parse import *
from parse_question import *
from preprocessing import *

restore = sys.argv[1]

#1. Setting
#batch_size
train_batch_size = valid_batch_size = 40
test_batch_size = 5

#input_layer
num_steps = 5
num_vocabulary = 12000

#dropout_layer
input_keep_prob = 1.0
output_keep_prob = 1.0

#rnn_layer
num_layers = 2
num_units = 256
forget_bias = 0.0

#optimizer
learning_rate = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-08

#train
num_epoch = 2

#2. Preprocessing
mypath = "./Holmes_Training_Data/"
logdir = "./save/"

train_datasets, train_labelsets = read_data(mypath, small=False)
test_datasets, test_optionsets = get_questions(), get_options()

words_ids = build_dictionary(train_datasets + train_labelsets, num_vocabulary)

train_datasets_id = label_id(train_datasets, words_ids)
train_labelsets_id = label_id(train_labelsets, words_ids)
test_datasets_id = label_id(test_datasets, words_ids)
test_optionset_id = label_id(test_optionsets, words_ids)

end_id = words_ids["<end>"]
set_size = len(train_datasets_id)

train_data = train_datasets_id[:(set_size//5)*4]
train_labels = train_labelsets_id[:(set_size//5)*4]

valid_data = train_datasets_id[(set_size//5)*4:]
valid_labels = train_labelsets_id[(set_size//5)*4:]

test_data = test_datasets_id
test_options = test_optionset_id


#3. Defining
#utilities
def one_hot(indices):
	return tf.one_hot(indices=indices, depth=num_vocabulary, on_value=1, off_value=0, axis=None, dtype=tf.int32)

#input_layer
def projection_layer(x):
	one_hot_p = one_hot(x)
	projection_weights = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary, num_units], name="projection_weights")
	product = tf.matmul(one_hot_p_reshape, projection_weights)
	return product
def embedding_layer(x):
	embedding_weights = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary, num_units], name="embedding_weights")
	embed = tf.nn.embedding_lookup(embedding_weights, x)
	return embed

#hidden_layer
def lstm_cell():
	return tf.contrib.rnn.BasicLSTMCell(num_units=num_units, forget_bias=forget_bias, state_is_tuple=True)
def cell_wrapper():
	return tf.contrib.rnn.DropoutWrapper(cell=lstm_cell(), input_keep_prob=input_keep_prob, output_keep_prob=output_keep_prob)
def multilayers():
    return tf.contrib.rnn.MultiRNNCell(cells=[ cell_wrapper() for _ in range(num_layers) ], state_is_tuple=True)

#output_layer
def softmax_layer(x):
	softmax_weights = tf.get_variable(dtype=tf.float32, shape=[num_units, num_vocabulary], name="softmax_weights")
	softmax_biases = tf.get_variable(dtype=tf.float32, shape=[num_vocabulary], name="softmax_biases")
	product = tf.add(tf.matmul(x, softmax_weights), softmax_biases)
	softmax_product = tf.nn.softmax(logits=product, dim=-1, name="softmax_product")
	return softmax_product

#loss
def sequence_loss_by_example(outputs, labels): 
	return tf.contrib.legacy_seq2seq.sequence_loss_by_example(
        	[outputs],
        	[tf.reshape(labels, [-1])],
        	[tf.ones([train_batch_size * num_steps], dtype=tf.float32)]
        )

#optimizer
def adam_optimizer(loss):
	return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon, use_locking=False).minimize(loss)

#4. Modeling
class Model:
	def __init__(self, input_layer_type, rnn_cell_type, output_layer_type, loss_type, optimizer_type, batch_size, is_testing = False):
		#placeholder
		self.x, self.y_ = tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name = "x"), tf.placeholder(dtype=tf.int32, shape=[None, num_steps], name="y_")
		#input_layer
		if(input_layer_type == "projection_layer"):
			self.input_layer = projection_layer(self.x)
		elif(input_layer_type == "embedding_layer"):
			self.input_layer = embedding_layer(self.x)
		#dropout_layer
		self.dropout_layer = tf.nn.dropout(self.input_layer, output_keep_prob)
		#rnn
		with tf.variable_scope("lstm"):
			if(rnn_cell_type == "lstm"):
				self.multilayers = multilayers()
				self.initial_state = self.multilayers.zero_state(batch_size, tf.float32)
				current_state = self.initial_state
				self.current_outputs = []
				for step in range(num_steps):
					if step > 0:
						tf.get_variable_scope().reuse_variables()
					current_output, current_state = self.multilayers(self.dropout_layer[:, step, :], current_state)
					self.current_outputs.append(current_output)
				self.final_state = current_state
				self.final_outputs = tf.reshape(tf.concat(axis=1, values=self.current_outputs), [-1, num_units])

		#output_layer
		if(output_layer_type == "softmax_layer"):
			self.output_layer = softmax_layer(self.final_outputs)
	
		if is_testing:
			return
			
		#loss
		if(loss_type == "sequence_loss_by_example"):
			self.loss = sequence_loss_by_example(self.output_layer, self.y_)
		#cost
		self.cost = tf.reduce_sum(self.loss) / batch_size
		#optimizer
		if(optimizer_type == "adam_optimizer"):
			self.optimizer = adam_optimizer(self.cost)

#5. Feeding
def feed_dict_to_model(session, model, is_training, data_batches, label_batches, num_batch, is_testing = False):
	total_cost_per_epoch = 0.0
	total_num_steps_per_epoch = 0
	initial_state = session.run(model.initial_state)
	if is_testing:
		fetch_dict = {
			"output_layer":model.output_layer,
			}
		answers_with_prod = []
		answers_with_sum = []
		for batch in range(num_batch):
			feed_dict = { model.x:data_batches[batch]}
			for i, (c, h) in enumerate(model.initial_state):
				feed_dict[c] = initial_state[i].c
				feed_dict[h] = initial_state[i].h
			track_dict = session.run(fetch_dict, feed_dict)
			outputs = track_dict["output_layer"]
			probs = []
			sums = []
			for idx in range(test_batch_size):
				prob = 1.0
				prob *= outputs[5*idx + 0, data_batches[batch, idx, 1]]
				prob *= outputs[5*idx + 1, data_batches[batch, idx, 2]]
				prob *= outputs[5*idx + 2, data_batches[batch, idx, 3]]
				prob *= outputs[5*idx + 3, data_batches[batch, idx, 4]]
				probs.append(prob)

				summ = 0
				summ += outputs[5*idx + 0, data_batches[batch, idx, 1]]
				summ += outputs[5*idx + 1, data_batches[batch, idx, 2]]
				summ += outputs[5*idx + 2, data_batches[batch, idx, 3]]
				summ += outputs[5*idx + 3, data_batches[batch, idx, 4]]
				sums.append(summ)
			
			answers_with_prod.append( str(chr(97 + np.argmax(probs))) )
			answers_with_sum.append( str(chr(97 + np.argmax(sums))) )
			# print outputs[:50,0]
			# print len(outputs[0])
			# print np.sum(outputs[0])
			# print len(outputs)
			# raw_input()
		f_out1 = open("ans_with_prod.csv", 'w') 
		f_out2 = open("ans_with_sum.csv", 'w') 
		f_out1.write("id,answer\n")
		f_out2.write("id,answer\n")
		for idx in range(len(answers_with_prod)):
			f_out1.write(str(idx+1)+","+answers_with_prod[idx]+"\n")	
			f_out2.write(str(idx+1)+","+answers_with_sum[idx]+"\n")	

		f_out1.close()
		f_out2.close()
		return

	fetch_dict = {
			"output_layer":model.output_layer,
			"loss":model.loss,
			"cost":model.cost
		}
		
	if(is_training):
		fetch_dict["optimizer"] = model.optimizer
	for batch in range(num_batch):
		feed_dict = { model.x:data_batches[batch], model.y_:label_batches[batch] }
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = initial_state[i].c
			feed_dict[h] = initial_state[i].h
		track_dict = session.run(fetch_dict, feed_dict)
		total_cost_per_epoch += track_dict["cost"]
		total_num_steps_per_epoch += num_steps
		if(batch % 1000 == 0):
			print "Batch: ", batch, "Perplexity: ", np.exp(total_cost_per_epoch / total_num_steps_per_epoch)
	return np.exp(total_cost_per_epoch / total_num_steps_per_epoch)

#6. Graphing
with tf.Graph().as_default():
	initializer = tf.random_uniform_initializer(dtype=tf.float32, minval=-0.1, maxval=0.1, seed=1234)
	with tf.name_scope("Train"):
		with tf.variable_scope("Model", reuse=None, initializer=initializer):
			train_model = Model(
					input_layer_type = "embedding_layer", 
					rnn_cell_type = "lstm", 
					output_layer_type = "softmax_layer", 
					loss_type = "sequence_loss_by_example", 
					optimizer_type = "adam_optimizer" ,
					batch_size = train_batch_size
				)
	with tf.name_scope("Valid"):
		with tf.variable_scope("Model", reuse=True, initializer=initializer):
			valid_model = Model(
					input_layer_type = "embedding_layer", 
					rnn_cell_type = "lstm", 
					output_layer_type = "softmax_layer", 
					loss_type = "sequence_loss_by_example", 
					optimizer_type = "adam_optimizer",
					batch_size = valid_batch_size
				)
	with tf.name_scope("Test"):
		with tf.variable_scope("Model", reuse=True, initializer=initializer):
			test_model = Model(
					input_layer_type = "embedding_layer", 
					rnn_cell_type = "lstm", 
					output_layer_type = "softmax_layer", 
					loss_type = "sequence_loss_by_example", 
					optimizer_type = "adam_optimizer" ,
					batch_size = test_batch_size,
					is_testing = True
				)
	saver = tf.train.Saver()
	init_op = tf.global_variables_initializer()
	with tf.Session() as session:
		if(restore == True):
			print("Model restored.")
			saver.restore(session, "./save/model.ckpt")
		else:
			session.run(init_op)
		for epoch in range(num_epoch):
			#training
			print "Now is training"
			train_data_batches, train_labels_batches, train_num_batch = generate_batches(train_data, train_labels, train_batch_size)
			average_cost_per_epoch = feed_dict_to_model(session, train_model, True, train_data_batches, train_labels_batches, train_num_batch)
			print "Epoch: ", epoch, "Perplexity: ", average_cost_per_epoch
			#validation
			print "Now is validation"
			valid_data_batches, valid_labels_batches, valid_num_batch = generate_batches(valid_data, valid_labels, valid_batch_size)
			average_cost_per_epoch = feed_dict_to_model(session, valid_model, False, valid_data_batches, valid_labels_batches, valid_num_batch)
			print "Epoch: ", epoch, "Perplexity: ", average_cost_per_epoch
		save_path = saver.save(session, "./save/model.ckpt")
  		print("Model saved in file: %s" % save_path)

  		print "TESTING DATA : "

		testing_data_batches, testing_num_batch = generate_testing_batches(test_data, test_options)
		feed_dict_to_model(session, test_model, False, testing_data_batches, None, testing_num_batch, is_testing = True)


	'''
	supervisor = tf.train.Supervisor(logdir=logdir, summary_op=None)
	with supervisor.managed_session() as session:
		for epoch in range(num_epoch):
			#training
			print "Now is training"
			train_data_batches, train_labels_batches, train_num_batch = generate_batches(train_data, train_labels, train_batch_size)
			average_cost_per_epoch = feed_dict_to_model(session, train_model, True, train_data_batches, train_labels_batches, train_num_batch)
			print "Epoch: ", epoch, "Perplexity: ", average_cost_per_epoch
			#validation
			print "Now is validation"
			valid_data_batches, valid_labels_batches, valid_num_batch = generate_batches(valid_data, valid_labels, valid_batch_size)
			average_cost_per_epoch = feed_dict_to_model(session, valid_model, False, valid_data_batches, valid_labels_batches, valid_num_batch)
			print "Epoch: ", epoch, "Perplexity: ", average_cost_per_epoch
			#testing
			print "Now is testing"
		supervisor.saver.save(session, logdir, global_step=supervisor.global_step)
	'''
