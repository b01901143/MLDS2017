#-*- coding: utf-8 -*-
import os
import operator
import collections
import pandas as pd
import numpy as np

def getInfo(info_path):
	return pd.read_csv(info_path, sep=",")

def buildVocab(label_sentences):
	temp_collection_list = []
	for sentence in label_sentences:
		temp_collection_list.extend(sentence.lower().split(" "))
	word_counts = collections.Counter(temp_collection_list)
	word_counts = sorted(word_counts.items(), key=operator.itemgetter(1), reverse=True)
	word_id = collections.OrderedDict()
	word_id["<PAD>"], word_id["<BOS>"], word_id["<EOS>"] = 0, 1, 2
	for index, pair in enumerate(word_counts):
		word_id[pair[0]] = index + 3
	id_word = { value : key for key, value in word_id.iteritems() }
	return word_id, id_word
