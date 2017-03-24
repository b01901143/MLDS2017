import os
import collections
import tensorflow as tf

num_vocabulary = 12000

def _read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().split()

def _build_vocab(filename, pretrained=None): 
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:num_vocabulary-1]
    #||volcab|| = 10000,  9999 for <unk>
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def _build_vocab_no_id(filename):
    data = _read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:num_vocabulary-1]
    #||volcab|| = 10000,  9999 for <unk>
    words, _ = list(zip(*count_pairs))
    words=words+('<unk>',)
    return words

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] if word in word_to_id else num_vocabulary-1 for word in data ] ### modified: 9999 for <unk>
def _file_to_word (filename, words):
	data = _read_words(filename)
	return [word if word in words else '<unk>' for word in data]

def _list_to_word_ids(data, word_to_id):
    return [word_to_id[word] if word in word_to_id else num_vocabulary-1 for word in data ] ### modified: 9999 for <unk>

def ptb_raw_data(data_path=None):
    print("collecting data")
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    word_to_id = _build_vocab(train_path)
    train_data_id = _file_to_word_ids(train_path, word_to_id)
    valid_data_id = _file_to_word_ids(valid_path, word_to_id)
    test_data_id = _file_to_word_ids(test_path, word_to_id)
    vocabulary_id = len(word_to_id)

	raw_data=[train_data_id, valid_data_id, test_data_id, word_to_id, vocabulary_id] 
    # f_out = open("log.txt" , "w")
    # for d in train_data:
    #     f_out.write(str(d)+" ")
    print("collecting data done")
    return raw_data

def ptb_producer(raw_data, batch_size, num_steps, name=None ):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 1) // num_steps
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=True).dequeue()
        x = tf.strided_slice(data, [0, i * num_steps], [batch_size, (i + 1) * num_steps])
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * num_steps + 1], [batch_size, (i + 1) * num_steps + 1])
        y.set_shape([batch_size, num_steps])
        return x, y, i

def ptb_producer_test(raw_data, batch_size, num_steps, name=None ):
    with tf.name_scope(name, "PTBProducer", [raw_data, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_data, name="raw_data", dtype=tf.int32)
        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])
        epoch_size = (batch_len - 0) // (num_steps+1)
        assertion = tf.assert_positive(epoch_size, message="epoch_size == 0, decrease batch_size or num_steps")
        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")
        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.strided_slice(data, [0, i * (num_steps+1)], [batch_size, (i + 1) * (num_steps+1)])[:,:5]
        x.set_shape([batch_size, num_steps])
        y = tf.strided_slice(data, [0, i * (num_steps+1)], [batch_size, (i + 1) * (num_steps+1)])[:,1:]
        y.set_shape([batch_size, num_steps])
        return x, y, i

# test = _build_vocab_no_id(os.path.join("./data/sets/cut/","train.txt"))
# print(test[0])
# print(test[1])
# print(test[9999])
# print(len(test))
# print (type(test))
