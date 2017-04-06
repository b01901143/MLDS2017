import tensorflow as tf
from parameter import *
from utility import *
from structure import *

raw_data = ptb_raw_data(data_path)
train_data, valid_data, test_data, word_to_id, _ = raw_data

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
    sv = tf.train.Supervisor(logdir=save_path)
    with sv.managed_session() as session:
        for i in range(num_epoch):
            session.run(train_model_figure.assign_new_lr, feed_dict={train_model_figure.new_lr: learning_rate * (learning_rate_decay ** max(i + 1 - learning_rate_decay_param, 0.0))})
            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(train_model_figure.lr)))
            train_perplexity = run_epoch(i,session, train_input_figure, train_model_figure, is_training=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(i,session, valid_input_figure, valid_model_figure)
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
        print("Saving model to %s." % save_path)
        sv.saver.save(session, save_path, global_step=sv.global_step)
