import tensorflow as tf

SCREEN_HEIGHT = 80
SCREEN_WIDTH = 80
SCREEN_LENGTH = 4
NUM_ACTIONS = 2
CONV_1_LAYER_SHAPE = [8, 8, 4, 32]
CONV_2_LAYER_SHAPE = [4, 4, 32, 64]
CONV_3_LAYER_SHAPE = [3, 3, 64, 64]
CONV_1_LAYER_STRIDES = [1, 4, 4, 1]
CONV_2_LAYER_STRIDES = [1, 2, 2, 1]
CONV_3_LAYER_STRIDES = [1, 1, 1, 1]
CONV_1_KERNEL_SHAPE = [1, 2, 2, 1]
CONV_1_KERNEL_STRIDES = [1, 2, 2, 1]
FC_1_LAYER_SHAPE = [1600, 512]
FC_2_LAYER_SHAPE = [512, 2]

def get_weight_and_bias(shape):	
	W = tf.get_variable(
		name="W",
		shape=shape,
		initializer=tf.truncated_normal_initializer(
			mean=0.0,
			stddev=0.01,
			dtype=tf.float32
		),
		regularizer=None,
		trainable=True,
		collections=None
	)
	b = tf.get_variable(
		name="b",
		shape=[shape[-1]],
		initializer=tf.constant_initializer(
			value=0.01,
			dtype=tf.float32
		),
		regularizer=None,
		trainable=True,
		collections=None
	)
	return W, b

class Agent:
	def __init__(self):
		with tf.variable_scope(
			name_or_scope="conv_1_layer",
			default_name=None,
			values=None,
			dtype=tf.float32,
			initializer=None,
			regularizer=None,
			reuse=False
		):
			self.conv_1_W, self.conv_1_b = get_weight_and_bias(
				shape=CONV_1_LAYER_SHAPE
			)
		with tf.variable_scope(
			name_or_scope="conv_2_layer",
			default_name=None,
			values=None,
			dtype=tf.float32,
			initializer=None,
			regularizer=None,
			reuse=False
		):
			self.conv_2_W, self.conv_2_b = get_weight_and_bias(
				shape=CONV_2_LAYER_SHAPE
			)
		with tf.variable_scope(
			name_or_scope="conv_3_layer",
			default_name=None,
			values=None,
			dtype=tf.float32,
			initializer=None,
			regularizer=None,
			reuse=False
		):
			self.conv_3_W, self.conv_3_b = get_weight_and_bias(
				shape=CONV_3_LAYER_SHAPE
			)
		with tf.variable_scope(
			name_or_scope="fc_1_layer",
			default_name=None,
			values=None,
			dtype=tf.float32,
			initializer=None,
			regularizer=None,
			reuse=False
		):
			self.fc_1_W, self.fc_1_b = get_weight_and_bias(
				shape=FC_1_LAYER_SHAPE
			)
		with tf.variable_scope(
			name_or_scope="fc_2_layer",
			default_name=None,
			values=None,
			dtype=tf.float32,
			initializer=None,
			regularizer=None,
			reuse=False
		):
			self.fc_2_W, self.fc_2_b = get_weight_and_bias(
				shape=FC_2_LAYER_SHAPE
			)									
	def build_agent(self):
		s_tensor = tf.placeholder(
			dtype=tf.float32,
			shape=(None, SCREEN_HEIGHT, SCREEN_WIDTH, SCREEN_LENGTH),
			name="s_tensor"
		)
		conv_1_h = tf.nn.relu(
			tf.nn.conv2d(
				input=s_tensor,
				filter=self.conv_1_W,
				strides=CONV_1_LAYER_STRIDES,
				padding="SAME",
				use_cudnn_on_gpu=None,
				data_format="NHWC",
				name="conv_1_h"
			) + self.conv_1_b
		)
		conv_1_p = tf.nn.max_pool(
			value=conv_1_h,
			ksize=CONV_1_KERNEL_SHAPE,
			strides=CONV_1_KERNEL_STRIDES,
			padding="SAME",
			data_format="NHWC",
			name="conv_1_p"
		)
		conv_2_h = tf.nn.relu(
			tf.nn.conv2d(
				input=conv_1_p,
				filter=self.conv_2_W,
				strides=CONV_2_LAYER_STRIDES,
				padding="SAME",
				use_cudnn_on_gpu=None,
				data_format="NHWC",
				name="conv_2_h"
			) + self.conv_2_b
		)
		conv_3_h = tf.nn.relu(
			tf.nn.conv2d(
				input=conv_2_h,
				filter=self.conv_3_W,
				strides=CONV_3_LAYER_STRIDES,
				padding="SAME",
				use_cudnn_on_gpu=None,
				data_format="NHWC",
				name="conv_3_h"
			) + self.conv_3_b
		)
		conv_3_h_flat = tf.reshape(conv_3_h, [-1, FC_1_LAYER_SHAPE[0]])
		fc_1_h = tf.nn.relu(
			tf.matmul(
				conv_3_h_flat, self.fc_1_W	
			) + self.fc_1_b
		)
		q_tensor = tf.matmul(
			fc_1_h, self.fc_2_W
		) + self.fc_2_b
		return s_tensor, q_tensor
