# generate vocab size small txt w/wo word2id dictionary and embedding dictionary


import numpy as np
from  os.path import  isfile
import sys
import pickle
vocab_size=sys.argv[2] if sys.argv[2]!='0' else 30000
embd_size = 300
glove_path=sys.argv[1] if sys.argv[1]!='0' else 'glove.42B.300d.txt'
glove_small_path = './glove.100k.txt'
def parse_word ( line ): # a word and embd_size vector
	_line=line.split()
	word = _line[0]
	_line= _line[1:]
	embd = np.float32(_line)
	return word , embd

def read_glove ():
    print(vocab_size)
    f_in = open(glove_path , 'r')
    lines = []
    for i in xrange(vocab_size-4):
        lines.append( f_in.readline() )
    if not isfile(glove_small_path):
		f_out=open(glove_small_path,'w')
		for i in xrange(vocab_size-4):
			f_out.write( lines[i])
		f_out.close()
    
    embd = np.zeros(shape=(vocab_size,embd_size),dtype=np.float32)
    wordtoix = {}
    ixtoword = {}

    ixtoword[0] = '<pad>'
    ixtoword[1] = '<bos>'
    ixtoword[2] = '<eos>'
    ixtoword[3] = '<unk>'

    wordtoix['<pad>'] = 0
    wordtoix['<bos>'] = 1
    wordtoix['<eos>'] = 2
    wordtoix['<unk>'] = 3

    for id in xrange(vocab_size-4):
		_word, _embd = parse_word(lines[id]) 
		wordtoix[_word] = id+4
		ixtoword[id+4]  = _word
		embd[id+4] = _embd

    embd[0:4] = np.random.standard_normal(size=(4,embd_size))
   
    word_dic =open('word_dic','wb')
    id_dic   =open('id_dic','wb')
    embd_dic =open('embd_dic','wb')
    pickle.dump(wordtoix,word_dic)
    pickle.dump(ixtoword,id_dic)
    pickle.dump(embd    ,embd_dic)
    word_dic.close()
    id_dic.close()
    embd_dic.close()
    f_in.close()
	
    print ('parse done.')
    return wordtoix, ixtoword , embd

read_glove()

test = pickle.load(open('embd_dic','r'))
print(test[4])