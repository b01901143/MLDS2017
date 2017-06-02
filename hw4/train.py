import os
import sys
import random
import tensorflow as tf
from parameter import * 
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from function import *


model_name="ver1"
def train():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, help="for example: seqGan1")
	parser.add_argument("--restore_version", type=str, help="for example: 300 ")
	parser.add_argument("--input_file", type=str, help="for example: ./data/training_question.txt")
	parser.add_argument("--output_file",type=str, help="for example: ./data/training_answer.txt")
	args = parser.parse_args()
	#model
	generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
	discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
	rollout = ROLLOUT(generator, 0.8)

	
	#makedirs
	
	if not os.path.exists(model_dir+model_name):
		os.makedirs(model_dir+model_name)
	if not os.path.exists(sample_dir+model_name):
		os.makedirs(sample_dir+model_name)
	random.seed(SEED)
	np.random.seed(SEED)
	assert START_TOKEN == 0

	gen_data_loader = Gen_Data_loader(BATCH_SIZE)
	likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
	vocab_size = 80000
	dis_data_loader = Dis_dataloader(BATCH_SIZE)


	
	#session and saver
	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	if restore_flag:
		saver.restore(session, model_dir +model_name + "/-" + restore_version)
	else:
		session.run(tf.global_variables_initializer())
	
    

	print '#########################################################################'
	print 'Start Adversarial Training...'
	for epoch in range(num_epoch):	
		for batch in range(len(sample_training_info_list)//batch_size):
			# Train the generator for one step
			
			for it in range(1):
				samples = generator.generate(sess)
				rewards = rollout.get_reward(sess, samples, 16, discriminator)
				feed = {generator.x: samples, generator.rewards: rewards}
				_ = sess.run(generator.g_updates, feed_dict=feed)
	
			# Test
			if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
				generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
				likelihood_data_loader.create_batches(eval_file)
				test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
				buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
				print 'total_batch: ', total_batch, 'test_loss: ', test_loss
	
			# Update roll-out parameters
			rollout.update_params()
	
			# Train the discriminator
			for _ in range(5):
				generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
				dis_data_loader.load_train_data(positive_file, negative_file)
	
				for _ in range(3):
					dis_data_loader.reset_pointer()
					for it in xrange(dis_data_loader.num_batch):
						x_batch, y_batch = dis_data_loader.next_batch()
						feed = {
							discriminator.input_x: x_batch,
							discriminator.input_y: y_batch,
							discriminator.dropout_keep_prob: dis_dropout_keep_prob
						}
						_ = sess.run(discriminator.train_op, feed)
	
			sys.stdout.write("\nEpochID: {0}, Saving Model...\n".format(epoch))
			saver.save(session, model_dir+model_name, global_step=epoch)


if __name__ == '__main__':
	train()
