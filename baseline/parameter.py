#vocabulary
num_vocabulary = 12000

#initializer
initial_scale = 0.1

#dropout_layer
input_keep_prob = 1.0
output_keep_prob = 1.0

#hidden_layer
num_units = 168
forget_bias = 0.0
num_layers = 1

#learning_rate
learning_rate = 1
learning_rate_decay = 0.5
learning_rate_decay_param = 3
max_grad_norm = 5

#batch, epoch
num_epoch = 10
train_batch_size = valid_batch_size = 128
train_num_steps = valid_num_steps = 20
test_batch_size = test_num_steps = 5
test_num_batch = 1040

#path
data_path = "./data/sets/Holmes/"
dump_path = "_".join((
		"num_vocabulary", str(num_vocabulary),
		"output_keep_prob", str(output_keep_prob),
		"num_units", str(num_units),
		"num_layers", str(num_layers),
		"learning_rate", str(learning_rate),
		"num_epoch", str(num_epoch),
		"train_batch_size", str(train_batch_size),
		"train_num_steps", str(train_num_steps)
	)) + "/"

#save_path = "./save/" + dump_path
#answer_path = "./answer/" + dump_path
save_path = "./save/baseline/"
answer_path = "./answer/baseline/"
