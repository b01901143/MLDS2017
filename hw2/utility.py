#-*- coding: utf-8 -*-
import pickle
import pandas as pd

def getInfo(info_path):
    return pd.read_csv(info_path, sep=",")

def loadDic(word_dic_path, id_dic_path, init_bias_dic_path, embed_dic_path):
	word_id = pickle.load(open(word_dic_path))
	id_word = pickle.load(open(id_dic_path))
	init_bias_vector = pickle.load(open(init_bias_dic_path))
	embd = pickle.load(open(embed_dic_path))
	return word_id, id_word, init_bias_vector, embd

def arr2str(words):
	string = ''
	for word in words:
		if word == "<eos>":
			break
		string += word + ' '	
	return string
