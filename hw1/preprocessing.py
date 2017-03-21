#packages
import sklearn
import collections
import numpy as np

#make_words_square
def make_words_square(file):
	sentence_list = file.readlines()
	words_square = [ sentence.strip("\n").split(" ")[0:5] for sentence in sentence_list ]
	return words_square

#make_datasets
def make_datasets(words_square):
	num_sentence = len(words_square)
	train_datasets = words_square[0:(num_sentence//5)*3]
	valid_datasets = words_square[(num_sentence//5)*3:(num_sentence//5)*4]
	test_datasets = words_square[(num_sentence//5)*4:]
	return num_sentence, train_datasets, valid_datasets, test_datasets

#square_to_list(helper)
def square_to_list(data_square):
	data_list = [ data_square[i][j] for i in range(len(data_square)) for j in range(len(data_square[i])) ]
	shape = (len(data_square), len(data_square[0]))
	return data_list, shape

#list_to_square(helper)
def list_to_square(data_list, shape):
	data_square = [ [ data_list[i*shape[1]+j] for j in range(shape[1]) ] for i in range(shape[0]) ]
	return data_square

#build_dictionary
def build_dictionary(train_datasets, num_vocabulary):
	train_datasets, _ = square_to_list(train_datasets)
	words_freq = collections.Counter(train_datasets).most_common(num_vocabulary-1)
	words_freq.append(("<unknown>", 0))
	words, _ = list(zip(*words_freq))
	words_ids = dict(zip(words, range(len(words))))
	return words_ids

#label_id
def label_id(datasets, words_ids):
	datasets_list, datasets_shape = square_to_list(datasets)
	datasets_id_list = [ words_ids[word] if word in words_ids else words_ids["<unknown>"] for word in datasets_list ]
	datasets_id = list_to_square(datasets_id_list, datasets_shape)
	return datasets_id

#generate_pairs
def generate_pairs(datasets_id, end_id):
	data = []
	labels = []
	num_data = len(datasets_id)
	for i in range(num_data):
		if(i == num_data - 1):
			break
		if(datasets_id[i][3] == end_id & datasets_id[i][4] == end_id):
			++i
			continue
		data.append(datasets_id[i])
		labels.append(datasets_id[i+1])
	assert(len(data) == len(labels))
	return data, labels, len(data)

#generate_batches
def generate_batches(data, labels, batch_size):
	assert(len(data) == len(labels))
	num_batch = len(data) // batch_size
	order = sklearn.utils.shuffle(range(batch_size * num_batch))
	data_array, labels_array, order_array = np.array(data), np.array(labels), np.array(order)
	data_array, labels_array = data_array[order_array], labels_array[order_array]
	data_batch, labels_batch = data_array.reshape(num_batch, batch_size, -1), labels_array.reshape(num_batch, batch_size, -1)

	return data_batch, labels_batch, num_batch

def generate_testing_batches(test_data, test_options):
	assert(len(test_data) == len(test_options))
	test_array = np.array(test_data, dtype=np.object_)
	test_array = np.repeat(test_array, 5, axis = 0)
	option_array = np.array(test_options, dtype=np.object_)

	for i in range(len(test_options)):
		test_array[5*i : 5*i+5, 2] = np.transpose(option_array[i])

	label_array = test_array[:,1:]
	test_array = test_array[:,:5]

	test_batch = test_array.reshape(len(test_options), 5, -1)
	label_batch = label_array.reshape(len(test_options), 5, -1)

	return test_batch, label_batch, len(test_options)



