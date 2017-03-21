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
class embd_table:
	def __init__(self):
		self._embd= embd
		self._word2id = word_dic
		#tf.global_variables_initializer().run()
	def lookupId(self, words):
		# return a list of word ids
                ids = []
                for w in words:
                    if w=='<start>':
                        ids.append(80000)
                    elif w=='<end>':
                        ids.append(90000)
                    else:
                        ids.append(self._word2id.get(w))
		return ids
	def lookupEmbd(self,ids):
		# return a list of embeddings
                embed=np.zeros(shape=(len(ids),200),dtype=np.float32)
                start_embd=np.full((200),0,dtype=np.float32)
                end_embd=np.full((200),1,dtype=np.float32)
                for idx, id in enumerate (ids):
                    if id== 80000:
                        embed[idx]=start_embd
                    elif id==90000:
                        embed[idx]=end_embd
                    else:
                        embed[idx]=self._embd[[id,0][id==None]]
                return embed

table=embd_table()
test=table.lookupId(['<start>','pig','<end>','is'])
print (test)
embed=table.lookupEmbd(test)
print (embed)
print (embed.shape)

