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
    initializer = tf.random_uniform_initializer(-initial_scale, initial_scale)
    with tf.name_scope("Train"):
        train_input_figure = InputFigure(train_data, train_batch_size, train_num_steps, "TrainInputFigure")
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            train_model_figure = ModelFigure(input_figure=train_input_figure, is_training=True)
        tf.summary.scalar("Training Loss", train_model_figure.cost)
        tf.summary.scalar("Learning Rate", train_model_figure.lr)
    with tf.name_scope("Valid"):
        valid_input_figure = InputFigure(valid_data, valid_batch_size, valid_num_steps, "ValidInputFigure")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            valid_model_figure = ModelFigure(input_figure=valid_input_figure, is_training=False)
        tf.summary.scalar("Validation Loss", valid_model_figure.cost)
    with tf.name_scope("Test"):
        test_input_figure = InputFigure(test_data, test_batch_size, test_num_steps, "TestInputFigure")
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            test_model_figure = ModelFigure(input_figure=test_input_figure, is_training=False)
    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as session:
        for i in range(num_epoch):
            session.run(train_model_figure.assign_new_lr, feed_dict={train_model_figure.new_lr: learning_rate * (learning_rate_decay ** max(i + 1 - learning_rate_decay_param, 0.0))})
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model_figure.lr)))
            train_perplexity = run_epoch(i,session, train_input_figure, train_model_figure, is_training=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(i,session, valid_input_figure, valid_model_figure)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            test_perplexity = run_epoch(i,session, test_input_figure, test_model_figure)
            print("Test Perplexity: %.3f" % test_perplexity)
        print("Saving model to %s." % save_path)
        sv.saver.save(session, save_path, global_step=sv.global_step)
