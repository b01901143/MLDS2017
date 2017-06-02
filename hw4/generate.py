import os
import sys
import random
import argparse
import tensorflow as tf
from parameter import * 
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
from utility import *



def generate():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_name", type=str, help="for example: seqGan1")
	parser.add_argument("--restore_version", type=str, help="for example: GAN-300 ")
	parser.add_argument("--model_type", type=str,default='SeqGAN' , help=" [S2S,RL,BEST] appended to output file name")
	parser.add_argument("--input_file", type=str, help="for example: ./data/testing_question.txt")
	parser.add_argument("--output_file",type=str, help="for example: ./data/testing_answer.txt")
	args = parser.parse_args()
	#model
	generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)

	
	#makedirs
	model_path  = model_dir+args.model_name
	sample_path = sample_dir+args.model_name
	if not os.path.exists(sample_path):
		os.makedirs(sample_path)
	random.seed(SEED)
	np.random.seed(SEED)
	assert START_TOKEN == 1


	gen_data_loader = Gen_Data_loader(BATCH_SIZE)
	#session and saver
	session = tf.InteractiveSession()
	saver = tf.train.Saver()
	saver.restore(session, model_path + '/' + args.restore_version)
    
  
	samples = generator.generate(sess)
	with open(output_file, 'wb') as out_file:
		out_file.write(samples)


if __name__ == '__main__':
	generate()