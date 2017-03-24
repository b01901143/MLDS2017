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

import pickle
import numpy as np

word2id = open('./word2id','rb')
id2embd = open('./id2embd','rb')

word_dic= pickle.load(word2id)
embd    = pickle.load(id2embd)
embd_size =200
class embd_table:
	def __init__(self):
		self._embd= embd
		self._word2id = word_dic
		self._word2id['<end>']=90000
		#tf.global_variables_initializer().run()
	def lookupId(self, words):
		return [word_dic[w] for w in words]
	def lookupEmbd(self,ids):
		# return a list of embeddings
		end_embd=np.full((embd_size),1,dtype=np.float32)
		embd = np.zeros(shape=[len(ids),embd_size], dtype=np.float32)
		for idx , id  in enumerate(ids):
			if id ==90000:
				embd[idx]=end_embd
			else:
				embd[idx]=self._embd[ [id,0][id==None] ]
			
		return embd
'''
table=embd_table()
test=table.lookupId(['no','pig','<end>','is'])
print (test)
embed=table.lookupEmbd(test)
print (embed)
print (embed.shape)
'''
