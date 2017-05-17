import tensorflow as tf

#layer
class batch_norm(object):
	def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
		with tf.variable_scope(name):
			self.epsilon = epsilon
			self.momentum = momentum
			self.ema = tf.train.ExponentialMovingAverage(decay=self.momentum)
			self.name = name

	def __call__(self, x, train=True):
		shape = x.get_shape().as_list()
		if train:
			with tf.variable_scope(self.name) as scope:
				self.beta = tf.get_variable("beta", [shape[-1]], initializer=tf.constant_initializer(0.))
				self.gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))
				try:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
				except:
					batch_mean, batch_var = tf.nn.moments(x, [0, 1], name='moments')	
				#ema_apply_op = self.ema.apply([batch_mean, batch_var])
				#self.ema_mean, self.ema_var = self.ema.average(batch_mean), self.ema.average(batch_var)
				#with tf.control_dependencies([ema_apply_op]):
				#	mean, var = tf.identity(batch_mean), tf.identity(batch_var)
				mean, var = tf.identity(batch_mean), tf.identity(batch_var)
		else:
			mean, var = self.ema_mean, self.ema_var
		normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, self.beta, self.gamma, self.epsilon, scale_after_normalization=True)
		return normed

def fullyConnectedLayer(input_feature, output_dim, name="fullyConnectedLayer"):
	with tf.variable_scope(name):
		weights = tf.get_variable(name="weights", shape=[input_feature.get_shape().as_list()[1], output_dim], dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.02))
		biases = tf.get_variable(name="biases", shape=[output_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		return tf.matmul(input_feature, weights) + biases

def convolutionLayer(input_feature, filter_shape, stride_shape, name="convolutionLayer"):
	with tf.variable_scope(name):
		weights = tf.get_variable(name="weights", shape=filter_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.02))
		biases = tf.get_variable(name="biases", shape=[filter_shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		z = tf.nn.conv2d(input_feature, weights, strides=stride_shape, padding="SAME")
		z = tf.reshape(tf.nn.bias_add(z, biases), z.get_shape())
		return z

def transConvolutionLayer(input_feature, filter_shape, stride_shape, output_shape, name="transConvolutionLayer"):
	with tf.variable_scope(name):
		weights = tf.get_variable(name="weights", shape=filter_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))
		biases = tf.get_variable(name="biases", shape=[output_shape[-1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
		z = tf.nn.conv2d_transpose(input_feature, weights, output_shape=output_shape, strides=stride_shape, padding="SAME")
		z = tf.reshape(tf.nn.bias_add(z, biases), z.get_shape())
		return z

#activation
def leakyReLU(x):
	return tf.maximum(x, 0.2*x)
