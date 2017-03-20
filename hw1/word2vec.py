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

import os
import sys
import threading
import time

from six.moves import xrange  # pylint: disable=redefined-builtin

import numpy as np
import tensorflow as tf
import embd.word2vec_optimized as w2v
save_path = './embd/embd_save/model.ckpt-2261065'
class embd_table:
	def __init__(self):
		"""Train a word2vec model."""
		opts = w2v.Options()
		config = tf.ConfigProto(allow_soft_placement=True)
		with tf.Graph().as_default(), tf.Session(config=config) as session:
			with tf.device("/cpu:0"):
				model = w2v.Word2Vec(opts, session)
				model.saver.restore(session, save_path)
				_embd_tensor =tf.nn.l2_normalize(model._w_in, 1)
				self._embd= _embd_tensor.eval()
				self._word2id = model._word2id
				self._id2word = model._id2word
		#tf.global_variables_initializer().run()
	def lookupId(self, words):
		# return a list of word ids
		return [self._word2id.get(w) for w in words]
	def embd(self,ids):
		# return a list of embeddings
		return self._embd[ids]
		

table=embd_table()
test=table.lookupId(['UNK','pig'])
print (test)
embed=table.embd(test)
print (embed)
print (embed.shape)


