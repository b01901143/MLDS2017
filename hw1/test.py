import sys
import numpy as np
import tensorflow as tf
from parameter import *
from utility import *
from structure import *

test_path = sys.argv[1]
answer_path = sys.argv[2]

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
        f_out = open(answer_path, 'w') 
        f_out.write("id,answer\n")  
        for idx in range(len(answers)):
            f_out.write(str(idx+1)+","+answers[idx]+"\n")   
        f_out.close()
    return np.exp(total_cost_per_epoch / total_num_steps_per_epoch)

raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, _ = raw_data
test_datasets, test_optionsets = get_questions(test_path), get_options(test_path)
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
