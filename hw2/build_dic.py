#packages
import os
import sys
import json
import codecs
import numpy as np

#argv
embed_size = sys.argv[1]
vocab_size = sys.argv[2]

#path
glove_path = "./glove/glove.6B." + sys.argv[1] + "d.txt"
train_feat_dir, train_label_dir, train_info_path = "./data/training/feat/", "./data/training/label/", "./data/training/info.csv"
test_feat_dir, test_label_dir, test_info_path = "./data/testing/feat/", "./data/testing/label/", "../data/testing/info.csv"
word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path = "word_dic", "id_dic", "init_bias_dic", "embed_dic"

#function
def getInfo(info_path):
    return pd.read_csv(info_path, sep=",")

def getFeat(feat_path):
    return np.load(feat_path)

def getLabel(label_path):
    return json.load(codecs.open(label_path, "r", "utf-8"))

def parseWord(line):
	_line=line.split()
	word = _line[0]
	_line= _line[1:]
	embed = np.float32(_line)
	return word , embed

def readAll():
    train_data, test_data = getInfo(train_info_path), getInfo(test_info_path)
    train_feats, train_labels = [ getFeat(train_feat_dir + path) for path in train_data["feat_path"].values ], [ getLabel(train_label_dir + path) for path in train_data["label_path"].values ]
    test_labels = [ getLabel(test_label_dir + path) for path in test_data["label_path"].values ]
    all_merge_labels = [ label for labels in train_labels for label in labels ] + [ label for labels in test_labels for label in labels ]
    return all_merge_labels

def buildVocab(label_sentences):
    temp_collection_list = []
    for sentence in label_sentences:
        temp_collection_list.extend(sentence.lower().split(" "))
    word_counts = collections.Counter(temp_collection_list)
    word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
    word_counts = word_counts[:vocab_size-3]
    word_counts.extend([("<pad>", len(label_sentences)), ("<bos>", len(label_sentences)), ("<eos>", len(label_sentences))])
    word_id = collections.OrderedDict()
    for index, pair in enumerate(word_counts):
        word_id[pair[0]] = index + 3
    word_id["<pad>"], word_id["<bos>"], word_id["<eos>"] = 0, 1, 2
    id_word = { value : key for key, value in word_id.iteritems() }
    init_bias_vector = np.array([ 1.0 * wc_tuple[1] for wc_tuple in word_counts ])
    init_bias_vector /= np.sum(init_bias_vector)
    init_bias_vector = np.log(init_bias_vector)
    init_bias_vector -= np.max(init_bias_vector)
    return word_id, id_word, init_bias_vector

def buildDic():
    #prepare data
    label_sentences = readAll()
    word_id, id_word, init_bias_vector = buildVocab() 
    #prepare glove
    embed_glove = {}
    embed = np.zeros(shape=(caption_size,embed_size) , dtype=np.float32)
    wordtoix, ixtoword = {}, {}
    ixtoword[0], ixtoword[1], ixtoword[2] = '<pad>', '<bos>' , '<eos>'
    wordtoix['<pad>'], wordtoix['<bos>'], wordtoix['<eos>'] = 0, 1, 2
    f_in = open(glove_path, 'r')
    lines = [ line for line in f_in.readlines() ]
    for id ,line in enumerate(lines):
        _word, _embed = parseWord(line)
        wordtoix [_word] = id+3
        embed_glove[id+3] = _embed
        print '\r parsing glove ', id ,
    for i in range(3):
        embed_glove[i] = np.random.standard_normal(size=embed_size)    
	#extract label's embedding
    for key, i in word_id.iteritems():
        ix = wordtoix.get(key)
        if ix is not None:
		    embed[i] = embed_glove[ix]
        else :
            embed[i] = np.random.standard_normal(size=embed_size)
        print(key, ' ', i, 'map to :', ix)	
    word_dic, id_dic, init_bias_dic, embed_dic = open('word_dic','wb'), open('id_dic','wb'), open('init_bias_dic', 'w') open('embed_dic','wb')
    pickle.dump(word_id, word_dic)
    pickle.dump(id_word, id_dic)
    pickle.dump(init_bias_vector, init_bias_dic)
    pickle.dump(embed, embed_dic)
    word_dic.close()
    id_dic.close()
    init_bias_dic.close()
    embed_dic.close()
    f_in.close()
    return wordtoix, ixtoword , embed

if __name__ == "__main__":
    buildDic()
