# generate vocab size small txt w/wo word2id dictionary and embedding dictionary

import sys
import os
from parameter import *
from utility import *
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
def read_labels ():
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_feats, train_labels = [ getFeat(train_feat_dir + path) for path in train_data["feat_path"].values ], [ getLabel(train_label_dir + path) for path in train_data["label_path"].values ]
    test_labels = [ getLabel(test_label_dir + path) for path in test_data["label_path"].values ]
    all_merge_labels = [ label for labels in train_labels for label in labels ] + [ label for labels in test_labels for label in labels ]
    return all_merge_labels
def write_small_glove( f_in ):
	for i in xrange(vocab_size-4):
		lines.append( f_in.readline() )
	if not os.path.isfile(glove_small_path):
		f_out=open(glove_small_path,'w')
		for i in xrange(vocab_size-4):
			f_out.write( lines[i])
		f_out.close()
def read_glove ():
    label_sentences=read_labels()
    temp_collection_list = []
    for sentence in label_sentences:
	    temp_collection_list.extend(sentence.lower().split(" "))
    word_counts = collections.Counter(temp_collection_list)
    word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
    word_counts.extend([("<pad>", len(label_sentences)), ("<bos>", len(label_sentences)), ("<eos>", len(label_sentences))])
    word_id = collections.OrderedDict()
    for index, pair in enumerate(word_counts):
	    word_id[pair[0]] = index + 3    
    word_id["<pad>"], word_id["<bos>"], word_id["<eos>"] = 0, 1, 2
    id_word = { value : key for key, value in word_id.iteritems() }

	#prepare glove
    embd_glove = {}
    embd = np.zeros(shape=(caption_size,embd_size) , dtype=np.float32)
    wordtoix = {}
    ixtoword = {}
    ixtoword[0] ,ixtoword[1] ,ixtoword[2]='<pad>', '<bos>' , '<eos>'
    #ixtoword[3] = '<unk>'
    wordtoix['<pad>'] ,wordtoix['<bos>'] ,wordtoix['<eos>'] = 0,1,2
    #wordtoix['<unk>'] = 3
    f_in = open(glove_path , 'r')
    lines = [line for line in f_in.readlines()]
    for id ,line in enumerate(lines):
        _word, _embd = parse_word(line)
        wordtoix [_word] = id+3
        embd_glove[id+3] = _embd
        print '\r parsing glove ', id ,
    for i in range(3):
        embd_glove[i] = np.random.standard_normal(size=embd_size)	   
	''' 
    for id in xrange(vocab_size-3):
		_word, _embd = parse_word(lines[id]) 
		wordtoix[_word] = id+3
		ixtoword[id+3]  = _word
		embd[id+3] = _embd
    '''
	# extract label's embedding

    for key, i in word_id.iteritems():
        ix=wordtoix.get(key)
        if ix is not None:
		    embd[i] = embd_glove[ix]
        else :
            embd[i] = np.random.standard_normal(size=embd_size)
        print(key,' ', i , 'map to :', ix )
	
    word_dic =open('word_dic','wb')
    id_dic   =open('id_dic','wb')
    embd_dic =open('embd_dic','wb')
    pickle.dump(word_id,word_dic)
    pickle.dump(id_word,id_dic)
    pickle.dump(embd ,embd_dic)
    word_dic.close()
    id_dic.close()
    embd_dic.close()
    f_in.close()
	
    print('check is:')
    print (embd[word_id['is']])
    print ('parse done.')
    return wordtoix, ixtoword , embd

read_glove()

test = pickle.load(open('embd_dic','r'))
print(test[4])