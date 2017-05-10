import os
import csv
import struct
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#download data and unzip them
mnist = input_data.read_data_sets("./data/", one_hot=True)
#==========mnist info==========
#mnist.train.images/labels, mnist.test.images/labels are arrays
#ndim(2, 2)
#shape((55000, 784), (10000, 784))
#size(43120000, 7840000)
#dtype(float32, float32)
#itemsize(4, 4)

#load test images
test_images_file_path = "./data/test-image"
test_images_file = open(test_images_file_path, "r")
magic, num, rows, columns = struct.unpack(">IIII", test_images_file.read(16))
test_images = np.fromfile(test_images_file, dtype="uint8").reshape(10000, rows * columns).astype("float32")

#construct session
session = tf.InteractiveSession()

#construct placeholder
x = tf.placeholder(tf.float32, shape=[None, 784])
y_label = tf.placeholder(tf.float32, shape=[None, 10])

#define layers
def weight_Variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_Variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv_layer(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")
def max_pool_layer(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#construct layers
x_image = tf.reshape(x, [-1, 28, 28, 1])
W_1 = weight_Variable([5, 5, 1, 32])
b_1 = bias_Variable([32])
conv_1 = tf.nn.relu(conv_layer(x_image, W_1) + b_1)
max_pool_1 = max_pool_layer(conv_1)

W_2 = weight_Variable([5, 5, 32, 64])
b_2 = bias_Variable([64])
conv_2 = tf.nn.relu(conv_layer(max_pool_1, W_2) + b_2)
max_pool_2 = max_pool_layer(conv_2)

W_fc_1 = weight_Variable([7 * 7 * 64, 1024])
b_fc_1 = bias_Variable([1024])
max_pool_flat_2 = tf.reshape(max_pool_2, [-1, 7 * 7 * 64])
a_fc_1 = tf.nn.relu(tf.matmul(max_pool_flat_2, W_fc_1) + b_fc_1)
keep_prob = tf.placeholder(tf.float32)
a_fc_drop_1 = tf.nn.dropout(a_fc_1, keep_prob)

W_fc_2 = weight_Variable([1024, 10])
b_fc_2 = bias_Variable([10])
a_fc_2 = tf.matmul(a_fc_drop_1, W_fc_2) + b_fc_2

#cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=a_fc_2))

#optimizer(gradient, update)
train = tf.train.AdamOptimizer(1e-4).minimize(cost)

#result
answer = tf.argmax(a_fc_2, 1)
comparison = tf.equal(tf.argmax(a_fc_2, 1), tf.argmax(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(comparison, tf.float32))

#run session
session.run(tf.global_variables_initializer())

#epoch
for i in range(36000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={ x: batch[0], y_label: batch[1], keep_prob: 1.0 })
        test_accuracy = accuracy.eval(feed_dict={ x: mnist.test.images, y_label: mnist.test.labels, keep_prob: 1.0 })
        print("epoch: %d, training_accuracy %g, testing_accuracy %g" % (i, train_accuracy, test_accuracy))
        #writefile
        test_answer = answer.eval(feed_dict={ x: test_images, keep_prob: 1.0})
        answer_dir = "./answer/"
        if not os.path.exists(answer_dir):
            os.mkdir(answer_dir)
        out_file = open(answer_dir + str(i) + ".csv", "wb")
        writer = csv.writer(out_file)
        writer.writerows([["id", "label"]])
        answer_list = []
        for j in range(10000):
            temp_list = []
            temp_list.append(str(j))
            temp_list.append(test_answer[j])
            answer_list.append(temp_list)
        writer.writerows(answer_list)
        out_file.close()
    train.run(feed_dict={ x: batch[0], y_label: batch[1], keep_prob: 0.5 })
