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



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="for example: seqGan1")
    parser.add_argument("--restore_version", type=str, help="for example: GAN-300 ")
    parser.add_argument("--input_file", type=str, help="for example: ./data/training_question.txt")
    parser.add_argument("--output_file",type=str, help="for example: ./data/training_answer.txt")
    args = parser.parse_args()
    #model
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    discriminator = Discriminator(sequence_length=2*SEQ_LENGTH, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)
    rollout = ROLLOUT(generator, 0.8)

    
    #makedirs
    model_path = model_dir+args.model_name
    sample_path= sample_dir+args.model_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 1

    # gen_data_loader = Gen_Data_loader(BATCH_SIZE)
    # likelihood_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    # dis_data_loader = Dis_dataloader(BATCH_SIZE)

    #data
    metadata, paired_data = get_paired_data()
    idx2w = metadata['idx2w']

    
    #session and saver
    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    if restore_flag:
        saver.restore(session, model_path + '/' + args.restore_version)
    else:
        session.run(tf.global_variables_initializer())
    
    #  pre-train generator
    # print 'Start pre-training...'
    # log.write('pre-training...\n')
    # for epoch in xrange(PRE_EPOCH_NUM):
    #     loss = pre_train_epoch(session, generator, gen_data_loader)
    #     if epoch % 5 == 0:
    #         generate_samples(session, generator, BATCH_SIZE, generated_num, eval_file)
    #         likelihood_data_loader.create_batches(eval_file)
    #         test_loss = target_loss(session, target_lstm, likelihood_data_loader)
    #         print 'pre-train epoch ', epoch, 'test_loss ', test_loss
    #         buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
    #         log.write(buffer)
    #    saver.save(session, model_path+'/pretrain_G', global_step=epoch)
    
    # print 'Start pre-training discriminator...'
    # # Train 3 epoch on the generated data and do this for 50 times
    # for _ in range(50):
    #     generate_samples(session, generator, BATCH_SIZE, generated_num, negative_file)
    #     dis_data_loader.load_train_data(positive_file, negative_file)
    #     for _ in range(3):
    #         dis_data_loader.reset_pointer()
    #         for it in xrange(dis_data_loader.num_batch):
    #             x_batch, y_batch = dis_data_loader.next_batch()
    #             feed = {
    #                 discriminator.input_x: x_batch,
    #                 discriminator.input_y: y_batch,
    #                 discriminator.dropout_keep_prob: dis_dropout_keep_prob
    #             }
    #             _ = session.run(discriminator.train_op, feed)
    #     saver.save(session, model_path+'/pretrain_D', global_step=_)
    print '#########################################################################'
    print 'Start Adversarial Training...'
    for epoch in range(TOTAL_EPOCH):
        shuffled_q,shuffled_a = shuffle_data(np.copy(paired_data))

        for batch in range(len(paired_data) // BATCH_SIZE):
            current_question = shuffled_q[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]
            current_answer = shuffled_a[batch * BATCH_SIZE : (batch+1) * BATCH_SIZE]


            # Train the generator for one step
            samples = generator.generate(session, current_question)

            rewards = rollout.get_reward(session, samples, 16, discriminator) #16
            feed = {generator.x: samples, generator.rewards: rewards, generator.question: current_question}
            _ = session.run(generator.g_updates, feed_dict=feed)

            # # Test
            if epoch % 10 == 0 and batch == 0:
                log = open('save/Epoch_' + str(epoch) + '.txt', 'w')
                log.write('Epoch : 'str(epoch) + '\n')
                for sete in samples:
                    log_list = []
                    for word in sete:
                        log_list.append(idx2w[word])
                        log.write( str(idx2w[word]) + ' ')
                    log.write('\n')
                log.close()

            # if total_batch % 5 == 0 or total_batch == TOTAL_BATCH - 1:
            #     generate_samples(session, generator, BATCH_SIZE, generated_num, eval_file)
            #     likelihood_data_loader.create_batches(eval_file)
            #     test_loss = target_loss(session, target_lstm, likelihood_data_loader)
            #     buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            #     print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            #     log.write(buffer)

            # Update roll-out parameters
            rollout.update_params()

            # Train the discriminator

            for _ in range(5):
                positive_sample = np.hstack((current_question,current_answer))
                negative_sample = generate_samples(session, generator, current_question, BATCH_SIZE, generated_num)
                x_batch = np.vstack((positive_sample,negative_sample))
                
                # Generate labels
                positive_labels = [[0, 1] for _ in positive_sample]
                negative_labels = [[1, 0] for _ in negative_sample]
                y_batch = np.concatenate([positive_labels, negative_labels], 0)

                for it in xrange(len(x_batch) // BATCH_SIZE):
                    feed = {
                        discriminator.input_x: x_batch[it * BATCH_SIZE : (it+1) * BATCH_SIZE],
                        discriminator.input_y: y_batch[it * BATCH_SIZE : (it+1) * BATCH_SIZE],
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _ = session.run(discriminator.train_op, feed)


if __name__ == '__main__':
    train()
