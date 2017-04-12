import os
import collections
import numpy as np
import pandas as pd
import tensorflow as tf
from parameter import *

def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().split()

def build_vocab(filename):
    data = read_words(filename)
    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    count_pairs = count_pairs[:num_vocabulary-1]
    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    return word_to_id

def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [ word_to_id[word] if word in word_to_id else num_vocabulary - 1 for word in data ]

def ptb_raw_data(data_path=None):
    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    word_to_id = build_vocab(train_path)
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, word_to_id, vocabulary

def ptb_producer(raw_data, batch_size, num_steps, name=None):
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

def ptb_producer_test(raw_data, batch_size, num_steps, name=None):
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

def generate_testing_batches(test_data, test_options):
    assert(len(test_data) == len(test_options))
    test_array = np.array(test_data, dtype=np.object_)
    test_array = np.repeat(test_array, 5, axis = 0)
    option_array = np.array(test_options, dtype=np.object_)
    for i in range(len(test_options)):
        test_array[5*i : 5*i+5, 2] = np.transpose(option_array[i])
    test_batch = test_array.reshape(len(test_options), 5, -1)
    return test_batch, len(test_options)

def get_questions(data_path):
    questions = []
    df = pd.read_csv(data_path)
    for item in df['question']:
        words = list(map(str,item.split()))
        i = 0
        while i < len(words):
            words[i] = words[i].lower()
            j = 0
            while j < len(words[i]):
                if not words[i][j].isalpha() and words[i] != '_____':
                    words[i] = words[i].replace(words[i][j], "")
                else:
                    j += 1
            if words[i] == "":
                del words[i]
            else:
                i += 1
        questions.append( words[:] + list(map(str,"<end> <end>".split())))
    for idx in range(0,len(questions)):
        for j in range(0,len(questions[idx])):
            if questions[idx][j] == '_____':
                questions[idx][j] = ' '
                questions[idx] = questions[idx][j-2 : j+4]
                break
    return questions

def get_options(data_path):
    df = pd.read_csv(data_path)
    options = []
    dat = list(map(list, df.values))
    for item in dat:
        options.append (item[2:])

        for i,word in enumerate(options[-1]):

            word = word.lower()
            j = 0
            while j < len(word):
                if not word[j].isalpha():
                    word = word.replace(word[j], "")
                else:
                    j += 1
            options[-1][i] = word
    return options
