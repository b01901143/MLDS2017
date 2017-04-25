#-*- coding: utf-8 -*-
import os
import operator
import collections
import pandas as pd
import numpy as np
import pickle
from parameter import  caption_size
from bleu import *
def getInfo(info_path):
	return pd.read_csv(info_path, sep=",")

def buildVocab(label_sentences):
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
	init_bias_vector = np.array([ 1.0 * wc_tuple[1] for wc_tuple in word_counts ])
	init_bias_vector /= np.sum(init_bias_vector)
	init_bias_vector = np.log(init_bias_vector)
	init_bias_vector -= np.max(init_bias_vector) 
	return word_id, id_word, init_bias_vector
def buildEmbd(label_sentences):
	word_id= pickle.load(open('../vocab/word_dic'))
	id_word= pickle.load(open('../vocab/id_dic'))
	embd   = pickle.load(open('../vocab/embd_dic'))
	init_bias_vector = np.zeros(caption_size,dtype=np.float32)
	return word_id , id_word , init_bias_vector ,embd
def arr2str(words):
	string=''
	for word in words:
		if word == "<eos>":
			break
		string+=word+' '	
	return string
def bleu_score(labels, caption):
	score = []
	for label in labels:
		score.append(BLEU_2(label[:-1], caption))
	score_mean = np.mean(score)
	print score_mean
	return score_mean
def inv_sigmoid(x):
	return np.log(1.0/x -1)
