# -*- coding:utf-8 -*-

import os
import sys
import time
import math
import reader
import random
import inspect
import collections
import numpy as np
import tensorflow as tf

reload(sys) 
sys.setdefaultencoding("utf-8")

os.environ["CUDA_VISIBLE_DEVICES"] = "7"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.95
session = tf.Session(config = config)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data-path", None, "Where the training/test data is stored.")
flags.DEFINE_string("vocab-path", None, "Where the wordlist file is stored.")
flags.DEFINE_string("save-path", None, "Model output directory.")
flags.DEFINE_integer("hidden-size", 200, "hidden dim of RNN")
flags.DEFINE_integer("num-layers", 2, "number of layers of RNN")
flags.DEFINE_integer("batch-size", 64, "batch size of RNN training")
flags.DEFINE_float("keep-prob", 1.0, "Keep Probability of Dropout")
flags.DEFINE_integer("max-epoch", 30, "The number of max epoch")


FLAGS = flags.FLAGS

class Config(object):
	"""Small config."""
	init_scale = 0.1
	learning_rate = 1.0
	max_grad_norm = 5
	num_layers = 2
	num_steps = 20
	hidden_size = 200
	max_epoch = 4
	max_max_epoch = 30
	keep_prob = 1.0
	lr_decay = 0.8
	batch_size = 64

# this new "softmax" function we show can train a "self-normalized" RNNLM where
# the sum of the output is automatically (close to) 1.0
# which saves a lot of computation for lattice-rescoring
def new_softmax(labels, logits):
	target = tf.reshape(labels, [-1])
	f_logits = tf.exp(logits)
	row_sums = tf.reduce_sum(f_logits, 1) # this is the negative part of the objf

	t2 = tf.expand_dims(target, 1)
	range = tf.expand_dims(tf.range(tf.shape(target)[0]), 1)
	ind = tf.concat([range, t2], 1)
	res = tf.gather_nd(logits, ind)

	return -res + row_sums - 1

class RnnlmInput(object):
	"""The input data."""
	def __init__(self, config, data, name=None):
		self.batch_size = batch_size = config.batch_size
		self.num_steps = num_steps = config.num_steps
		self.epoch_size = ((len(data) // batch_size) - 1) // num_steps
		self.input_data, self.targets = reader.rnnlm_producer(
			data, batch_size, num_steps, name=name)

class RnnlmModel(object):
	"""The RNNLM model."""
	def __init__(self, is_training, config, input_, skipgram_embeddings):
		self._input = input_
		batch_size = input_.batch_size
		num_steps = input_.num_steps
		hidden_size = config.hidden_size
		vocab_size = config.vocab_size

		def lstm_cell():
			# With the latest TensorFlow source code (as of Mar 27, 2017),
			# the BasicLSTMCell will need a reuse parameter which is unfortunately not
			# defined in TensorFlow 1.0. To maintain backwards compatibility, we add
			# an argument check here:
			if 'reuse' in inspect.getargspec(tf.contrib.rnn.BasicLSTMCell.__init__).args:
				return tf.contrib.rnn.BasicLSTMCell(
					hidden_size, forget_bias = 0.0, state_is_tuple = True,
					reuse = tf.get_variable_scope().reuse)
			else:
				return tf.contrib.rnn.BasicLSTMCell(
					hidden_size, forget_bias = 0.0, state_is_tuple = True)

		attn_cell = lstm_cell
		if is_training and config.keep_prob < 1:
			def attn_cell():
				return tf.contrib.rnn.DropoutWrapper(
					lstm_cell(), output_keep_prob = config.keep_prob)
		self.cell = tf.contrib.rnn.MultiRNNCell(
			[attn_cell() for _ in range(config.num_layers)], 
			state_is_tuple = True)

		self._initial_state = self.cell.zero_state(batch_size, tf.float32)
		self._initial_state_single = self.cell.zero_state(1, tf.float32)

		self.initial = tf.reshape(tf.stack(axis = 0, values = self._initial_state_single),
			[config.num_layers, 2, 1, hidden_size], name = "test_initial_state")

		# first implement the less efficient version
		test_word_in = tf.placeholder(tf.int32, [1, 1], name="test_word_in")

		state_placeholder = tf.placeholder(tf.float32, 
			[config.num_layers, 2, 1, hidden_size], name = "test_state_in")

		# unpacking the input state context 
		l = tf.unstack(state_placeholder, axis=0)
		test_input_state = tuple(
			[tf.contrib.rnn.LSTMStateTuple(l[idx][0],l[idx][1])
			 for idx in range(config.num_layers)]
		)

		self.embedding = tf.constant(skipgram_embeddings, shape = [vocab_size, hidden_size], dtype = tf.float32)
		inputs = tf.nn.embedding_lookup(self.embedding, input_.input_data)
		test_inputs = tf.nn.embedding_lookup(self.embedding, test_word_in)

		# test time
		with tf.variable_scope("RNN"):
			(test_cell_output, test_output_state) = self.cell(test_inputs[:, 0, :], test_input_state)

		test_state_out = tf.reshape(tf.stack(axis = 0, values = test_output_state), 
			[config.num_layers, 2, 1, hidden_size], name = "test_state_out")

		test_cell_out = tf.reshape(test_cell_output, [1, hidden_size], name = "test_cell_out")
		# above is the first part of the graph for test
		# test-word-in
		#               > ---- > test-state-out
		# test-state-in        > test-cell-out

		# below is the 2nd part of the graph for test
		# test-word-out
		#               > prob(word | test-word-out)
		# test-cell-in

		test_word_out = tf.placeholder(tf.int32, [1, 1], name = "test_word_out")
		cellout_placeholder = tf.placeholder(tf.float32, [1, hidden_size], name = "test_cell_in")

		softmax_w = tf.get_variable("softmax_w", [hidden_size, vocab_size], dtype = tf.float32)
		softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype = tf.float32)
		softmax_b = softmax_b - 9.0

		test_logits = tf.matmul(cellout_placeholder, 
			tf.transpose(tf.nn.embedding_lookup(tf.transpose(softmax_w), test_word_out[0]))) + softmax_b[test_word_out[0,0]]

		p_word = test_logits[0, 0]
		test_out = tf.identity(p_word, name = "test_out")

		if is_training and config.keep_prob < 1:
			inputs = tf.nn.dropout(inputs, config.keep_prob)

		# Simplified version of models/tutorials/rnn/rnn.py's rnn().
		# This builds an unrolled LSTM for tutorial purposes only.
		# In general, use the rnn() or state_saving_rnn() from rnn.py.
		#
		# The alternative version of the code below is:
		#
		# inputs = tf.unstack(inputs, num=num_steps, axis=1)
		# outputs, state = tf.contrib.rnn.static_rnn(
		#     cell, inputs, initial_state=self._initial_state)
		outputs = []
		state = self._initial_state
		with tf.variable_scope("RNN"):
			for time_step in range(num_steps):
				if time_step > -1: tf.get_variable_scope().reuse_variables()
				(cell_output, state) = self.cell(inputs[:, time_step, :], state)
				outputs.append(cell_output)

		output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, hidden_size])
		logits = tf.matmul(output, softmax_w) + softmax_b
		loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
				[logits],
				[tf.reshape(input_.targets, [-1])],
				[tf.ones([batch_size * num_steps], dtype=tf.float32)],
				softmax_loss_function = new_softmax)
		self._cost = cost = tf.reduce_sum(loss) / batch_size
		self._final_state = state

		if not is_training:
			return

		self._lr = tf.Variable(0.0, trainable=False)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
		optimizer = tf.train.GradientDescentOptimizer(self._lr)
		self._train_op = optimizer.apply_gradients(
				zip(grads, tvars),
				global_step=tf.contrib.framework.get_or_create_global_step())

		self._new_lr = tf.placeholder(
				tf.float32, shape=[], name="new_learning_rate")
		self._lr_update = tf.assign(self._lr, self._new_lr)

	def assign_lr(self, session, lr_value):
		session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

	@property
	def input(self):
		return self._input

	@property
	def initial_state(self):
		return self._initial_state

	@property
	def cost(self):
		return self._cost

	@property
	def final_state(self):
		return self._final_state

	@property
	def lr(self):
		return self._lr

	@property
	def train_op(self):
		return self._train_op

def run_epoch(session, model, eval_op=None, verbose=False):
	"""Runs the model on the given data."""
	start_time = time.time()
	costs = 0.0
	iters = 0
	state = session.run(model.initial_state)

	fetches = {
		"cost": model.cost,
		"final_state": model.final_state,
	}
	if eval_op is not None:
		fetches["eval_op"] = eval_op

	for step in range(model.input.epoch_size):
		feed_dict = {}
		for i, (c, h) in enumerate(model.initial_state):
			feed_dict[c] = state[i].c
			feed_dict[h] = state[i].h

		vals = session.run(fetches, feed_dict)
		cost = vals["cost"]
		state = vals["final_state"]

		costs += cost
		iters += model.input.num_steps

		if verbose and step % (model.input.epoch_size // 10) == 10:
			print("%.3f perplexity: %.3f speed: %.0f wps" %
				(step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
				 iters * model.input.batch_size / (time.time() - start_time)))

	return np.exp(costs / iters)

data_index = 0
def generate_batch(train_data, embed_batch_size, num_skips, skip_window):
	global data_index
	assert embed_batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window

	batch = np.ndarray(shape = (embed_batch_size), dtype = np.int32)
	labels = np.ndarray(shape = (embed_batch_size, 1), dtype = np.int32)

	span = 2 *  skip_window + 1
	buffer = collections.deque(maxlen = span)

	for _ in range(span):
		buffer.append(train_data[data_index])
		data_index = (data_index + 1) % len(train_data)

	for i in range(embed_batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(train_data[data_index])
		data_index = (data_index + 1) % len(train_data)
	return batch, labels

def get_config():
	return Config()

def main(_):
	if not FLAGS.data_path:
		raise ValueError("Must set --data_path to RNNLM data directory")

	raw_data = reader.rnnlm_raw_data(FLAGS.data_path, FLAGS.vocab_path)
	train_data, valid_data, _, word_map = raw_data
	reverse_wordmap = dict(zip(word_map.values(), word_map.keys()))
	vocabulary_size = len(word_map)

	config = get_config()
	config.vocab_size = len(word_map)
	config.hidden_size = FLAGS.hidden_size
	config.num_layers = FLAGS.num_layers
	config.batch_size = FLAGS.batch_size
	config.keep_prob= FLAGS.keep_prob
	config.max_max_epoch = FLAGS.max_epoch
	
	eval_config = get_config()
	eval_config.batch_size = 1
	eval_config.num_steps = 1

	# word embedding参数设置
	embed_batch_size = 128
	embedding_size = 100
	skip_window = 1
	num_skips = 2

	valid_size = 16
	valid_window = 100
	embed_num_steps = 100001

	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(
		valid_examples,
		random.sample(range(1000, 1000 + valid_window),valid_size // 2))
	num_sampled = 64

	graph_skipgram = tf.Graph()
	with graph_skipgram.as_default():
		train_dataset = tf.placeholder(tf.int32, shape = [embed_batch_size])
		train_labels = tf.placeholder(tf.int32, shape = [embed_batch_size, 1])
		valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

		embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
		softmax_weights = tf.Variable(
			tf.truncated_normal([vocabulary_size, embedding_size],
				stddev = 1.0/math.sqrt(embedding_size)))
		softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

		embed = tf.nn.embedding_lookup(embeddings, train_dataset)
		print("Embed size: %s" % embed.get_shape().as_list())

		loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
			weights = softmax_weights,
			biases = softmax_biases, 
			inputs = embed, 
			labels = train_labels, 
			num_sampled = num_sampled, 
			num_classes = vocabulary_size))

		optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims = True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))

	with tf.Session(graph = graph_skipgram) as session:
		tf.global_variables_initializer().run()
		print("Initialized!")
		average_loss = 0
		for step in range(embed_num_steps):
			batch_data, batch_labels = generate_batch(
				train_data = train_data,
				embed_batch_size = embed_batch_size,
				num_skips = num_skips,
				skip_window = skip_window)
			feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
			_, lo = session.run([optimizer, loss], feed_dict = feed_dict)
			average_loss += lo
			if step % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				print("Averge loss at step %d: %f" % (step, average_loss))
				average_loss = 0
			if step % 10000 == 0:
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = reverse_wordmap[valid_examples[i]]
					top_k = 8
					nearest = (-sim[i,:]).argsort()[1:top_k+1]
					log = "Nearest to %s:" % valid_word
					for k in range(top_k):
						close_word = reverse_wordmap[nearest[k]]
						log = log + " " + close_word + ","
					print(log)
		final_embeddings = normalized_embeddings.eval()

	with tf.Graph().as_default():
		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

		with tf.name_scope("Train"):
			train_input = RnnlmInput(config = config, data = train_data, name = "TrainInput")
			with tf.variable_scope("Model", reuse = None, initializer = initializer):
				m = RnnlmModel(is_training = True, config = config, input_ = train_input,
					skipgram_embeddings = final_embeddings)
			tf.summary.scalar("Training Loss", m.cost)
			tf.summary.scalar("Learning Rate", m.lr)

		with tf.name_scope("Valid"):
			valid_input = RnnlmInput(config = config, data = valid_data, name = "ValidInput")
			with tf.variable_scope("Model", reuse = True, initializer = initializer):
				mvalid = RnnlmModel(is_training = False, config = config, input_ = valid_input,
					skipgram_embeddings = final_embeddings)
			tf.summary.scalar("Validation Loss", mvalid.cost)

		sv = tf.train.Supervisor(logdir=FLAGS.save_path)
		with sv.managed_session() as session:
			for i in range(config.max_max_epoch):
				lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
				m.assign_lr(session, config.learning_rate * lr_decay)

				print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
				train_perplexity = run_epoch(session, m, eval_op=m.train_op, verbose=True)

				print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
				valid_perplexity = run_epoch(session, mvalid)
				print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

			if FLAGS.save_path:
				print("Saving model to %s." % FLAGS.save_path)
				sv.saver.save(session, FLAGS.save_path)

if __name__ == "__main__":
	tf.app.run()
