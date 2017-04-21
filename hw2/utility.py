#-*- coding: utf-8 -*-
import os
import json
import codecs
import operator
import collections
import pandas as pd
import numpy as np
import pickle

def getInfo(info_path):
	return pd.read_csv(info_path, sep=",")

def getFeat(feat_path):
	return np.load(feat_path)

def getLabel(label_path):
	return json.load(codecs.open(label_path, "r", "utf-8"))

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
def buildEmbd():
	word_id= pickle.load(open('word_dic'))
	id_word= pickle.load(open('id_dic'))
	embd   = pickle.load(open('embd_dic'))
	return word_id , id_word , embd