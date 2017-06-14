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
from data_parser import *



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

    metadata, paired_data = get_paired_data()
    idx2w = metadata['idx2w']


    gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    #session and saver
    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(session, model_path + '/' + args.restore_version)

    shuffled_q, shuffled_a = shuffle_data(np.copy(paired_data))
    num_batch = len(shuffled_q) // BATCH_SIZE
    _time = time.time()

    total_samples = []
    train_samples = []

    for it in xrange(num_batch):

        current_question = shuffled_q[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
        current_answer = shuffled_a[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
        batch = np.hstack((current_question,current_answer))
        samples = generator.generate_test(session, current_question)

        if it == 0:
            total_samples = samples
            train_samples = batch

        else:
            total_samples = np.vstack((total_samples,samples))
            train_samples = np.vstack((train_samples,batch))

    # save_samples(total_samples ,idx2w=idx2w , sample_path=sample_path+str(epoch)+'.txt')
    save_samples(total_samples ,idx2w=idx2w , sample_path='testing.txt')
    save_samples(train_samples ,idx2w=idx2w , sample_path='training_data.txt')




if __name__ == '__main__':
    generate()
