import numpy as np
import tensorflow as tf
from parameter import *
from utility import *
from structure import *

raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, _ = raw_data
test_datasets, test_optionsets = get_questions(), get_options()
testing_data_batches, testing_num_batch = generate_testing_batches(test_datasets, test_optionsets)
for i in range(len(testing_data_batches)):
    for j in range(len(testing_data_batches[i])):
        testing_data_batches[i,j] = [word_to_id[word] if word in word_to_id else num_vocabulary-1 for word in testing_data_batches[i,j] ]
test_data = np.reshape( np.swapaxes(testing_data_batches, 0, 1), -1 )

with tf.Graph().as_default():
    initializer = tf.local_variables_initializer()
    with tf.name_scope("Test"):
        test_input_figure = InputFigure(test_data, test_batch_size, test_num_steps, "TestInputFigure")
        with tf.variable_scope("Model", reuse=None):
            test_model_figure = ModelFigure(input_figure=test_input_figure, is_training=False)
    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as session:
        session.run(initializer)
        test_perplexity = run_epoch(i, session, test_input_figure, test_model_figure)
        print("Test Perplexity: %.3f" % test_perplexity)
