# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Multi-threaded word2vec unbatched skip-gram model.

Trains the model described in:
(Mikolov, et. al.) Efficient Estimation of Word Representations in Vector Space
ICLR 2013.
http://arxiv.org/abs/1301.3781
This model does true SGD (i.e. no minibatching). To do this efficiently, custom
ops are used to sequentially process data within a 'batch'.

The key ops used are:
* skipgram custom op that does input processing.
* neg_train custom op that efficiently calculates and applies the gradient using
  true SGD.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from  reader import num_vocabulary
import pickle
import tensorflow as tf
import numpy as np

word2id = open('./word2id','rb')
id2embd = open('./id2embd','rb')

word_dic= pickle.load(word2id)
embd    = pickle.load(id2embd)
vocab_size=71292
embd_size =200
class embd_table:
	def __init__(self):
		self._word2id = word_dic
		
		self._word2id['<end>']=71291
		self.embed=np.append(embd,[np.full((embd_size),0.0,dtype=np.float32)],axis=0)
		print(type(word_dic))
		print(type(self._word2id))
		print (embd.shape)
		print (self.embed.shape)
		#print(self._embd.shape)
		#tf.global_variables_initializer().run()
	def lookupId(self, word):
		return self._word2id[word]
	def lookupEmbd(self,word_2_id):
		# return a list of embeddings
		_embedding = np.zeros(shape=[num_vocabulary,embd_size],dtype=np.float32)
		for key , value in word_2_id.iteritems():
			_embedding[value]=self.embed[self.lookupId(key)]
			'''
			print(self.lookupId(key))
			print (key)
			print (value)
			print(_embedding[value])
			'''
		embd = tf.constant(_embedding,shape=[num_vocabulary,embd_size], dtype=tf.float32)
	
			
		return embd


table=embd_table()
#test=table.lookupId(['no','pig','<end>','is'])
#print (test)
embed=table.lookupEmbd({'pig':0,'<end>':1})
'''
with tf.Session() as sess:
	tf.global_variables_initializer().run()
	_=embed.eval()
	print(_[1])
	print(_.shape)
'''
