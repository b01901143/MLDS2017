import numpy as np
import random
import pickle
import time , sys ,os
from parameter import *

glove_txt_path = '../vocab/glove.42B.300d.txt'
embed_path  = './data/glove_300d.pic'
def generate_samples(sess, trainable_model, current_question, batch_size, generated_num):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, current_question))

    return np.asarray(generated_samples, dtype = np.int32)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.pretrain_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, shuffled_q, shuffled_a):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []

    num_batch = len(shuffled_q) // BATCH_SIZE

    for it in xrange(num_batch):
    	current_question = shuffled_q[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
        current_answer = shuffled_a[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
    	batch = np.hstack((current_question,current_answer))

        _, g_loss = trainable_model.pretrain_step(sess, batch, current_question)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
	
def build_glove_embed(id2word):
	def parseWord(line):
		_line=line.split()
		word = _line[0]
		_line= _line[1:]
		embed = np.float32(_line)
		return word , embed
	embed = np.random.standard_normal(size=(vocab_size,EMB_DIM) )
	embed_glove ,wordtoix, ixtoword = {}, {} , {}
	
	ixtoword[0], ixtoword[1], ixtoword[2] = '<pad>', '<bos>' , '<eos>'
	wordtoix['<pad>'], wordtoix['<bos>'], wordtoix['<eos>'] = 0, 1, 2
	f_in = open(glove_txt_path, 'r')
	print "loading glove"
	lines = [ line for line in f_in.readlines() ]
	for id ,line in enumerate(lines):
		_word, _embed = parseWord(line)
		wordtoix [_word] = id + 3
		embed_glove[id+3] = _embed
		if id % 100 == 0:
			print "parsing glove: " + str(id) +'\r',
			sys.stdout.flush()

	
	#extract label's embedding
	for i, key in enumerate(id2word):
		ix = wordtoix.get(key)
		if ix is not None:
			embed[i] = embed_glove[ix]
		if id % 100 == 0:
			print key + ' '+str(i), "is map to :" + str(ix) + '\r' ,
			sys.stdout.flush()
	if not os.path.exists('./data'):
		os.makedirs('./data')
	print "dumping"
	pickle.dump(embed, open(embed_path,'wb'))
	return None
def embed_initilize():
	embed = pickle.load(open(embed_path,'r'))
	embed = tf.get_variable(embed)
	return embed

if __name__ == '__main__':
	dic = ['is','happy']
	build_glove_embed(dic)
	
