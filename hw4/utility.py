import numpy as np
import random
import pickle
import io
import time , sys ,os
from parameter import *

glove_txt_path = '../vocab/glove.42B.300d.txt'
embed_path  = './data/glove_300d.pic'
UNK = '<unk>'
def generate_samples(sess, trainable_model, current_question, batch_size, generated_num):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess, current_question))

    return np.asarray(generated_samples, dtype = np.int32)
def save_samples(current_answer,idx2w , sample_path):
    def id2sen (sentence , dic = idx2w):
        s="Q: " 
        for id in sentence:
            w = dic[id]
            if w == '<eoq>':
                s+="A: "
                continue
            if w != '<pad>' and w!='<unk>':
                s+=( w +' ')
        return s
    
    QA = [ id2sen(sentence)for sentence in current_answer]
    with open(sample_path,'w') as f:
        for qa in QA:

            f.write(qa+'\n')

def save_test_samples(current_answer,idx2w , sample_path):
    def id2sen (sentence , dic = idx2w):
        s="" 
        start = False
        for id in sentence:
            w = dic[id]
            if w == '<eoq>':
                start = True
                continue
            if w != '<pad>' and w!='<unk>' and start:
                s+=( w +' ')
        return s
    
    QA = [ id2sen(sentence)for sentence in current_answer]
    with open(sample_path,'w') as f:
        for qa in QA:

            f.write(qa+'\n')

def zero_pad(qtokenized, w2idx):
    # num of rows
    data_len = len(qtokenized)

    # numpy arrays to store indices
    idx_q = np.zeros([data_len, 20], dtype=np.int32)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, 20)

        idx_q[i] = np.array(q_indices)

        idx_q[i][-1] = w2idx['<eoq>']

    return idx_q

def pad_seq(seq, lookup, maxlen):
    indices = []
    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])
    return indices + [0]*(maxlen - len(seq))

def get_sample_input(w2idx, sample_path = 'sample_input.txt'):
    qtokenized =[]
    lines = io.open(sample_path, encoding='utf-8', errors='ignore').read().split('\n')
    for line in lines:
        qtokenized.append(line.split(' '))

    idx_q = zero_pad(qtokenized, w2idx)

    return idx_q

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


def pre_train_epoch(sess, trainable_model, shuffled_q, shuffled_a ):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []

    num_batch = len(shuffled_q) // BATCH_SIZE
    _time = time.time()
    for it in xrange(num_batch):
        
        current_question = shuffled_q[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
        current_answer = shuffled_a[it * BATCH_SIZE : (it+1) * BATCH_SIZE]
        batch = np.hstack((current_question,current_answer))
        _, g_loss = trainable_model.pretrain_step(sess, batch, current_question)
        supervised_g_losses.append(g_loss)
        if it % 100 == 0:
            
            print "batch:"+str(it)+" loss:"+str(g_loss)+" time taken:"+str(int(time.time()-_time))+'        \r',
            sys.stdout.flush()
            _time = time.time()
     

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
class Timer:
    def __init__ (self):
        self.last = [time.time() for _ in range(5) ]
        self.duration = [ 0.0 for _ in range(5) ]
    def refresh(self , id ):
        self.duration[id]= time.time()-self.last[id]
        self.last[id] = time.time()
    def count (self,id):
        self.refresh(id)
        return str(int(self.duration[id]))
if __name__ == '__main__':
    dic = ['is','happy']
    build_glove_embed(dic)
    
