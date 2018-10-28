# -*- coding:utf-8 -*-

import os
import sys
import math
import reader
import random
import collections
import numpy as np
import tensorflow as tf

reload(sys) 
sys.setdefaultencoding("utf-8")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config = config)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data-path", None, "the path to train data")
flags.DEFINE_string("vocab-path", None, "the path to the vocab")
flags.DEFINE_string("ckpt-path", None, "the path to the checkpoint")
flags.DEFINE_integer("embedding-size", 200, "embedding dim of RNN")
flags.DEFINE_integer("batch-size", 128, "batch size of word embedding")

FLAGS = flags.FLAGS

raw_data = reader.rnnlm_raw_data(FLAGS.data_path, FLAGS.vocab_path)
train_data, valid_data, _, word_map = raw_data
reverse_wordmap = dict(zip(word_map.values(), word_map.keys()))
vocabulary_size = len(word_map)

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window

	batch = np.ndarray(shape = (batch_size), dtype = np.int32)
	labels = np.ndarray(shape = (batch_size, 1), dtype = np.int32)

	span = 2 *  skip_window + 1
	buffer = collections.deque(maxlen = span)

	for _ in range(span):
		buffer.append(train_data[data_index])
		data_index = (data_index + 1) % len(train_data)

	for i in range(batch_size // num_skips):
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

num_steps = 100001
if __name__ == '__main__':

	batch_size = FLAGS.embedding_size
	embedding_size = FLAGS.embedding_size
	skip_window = 1
	num_skips = 2

	valid_size = 16
	valid_window = 100
	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(
		valid_examples,
		random.sample(range(1000, 1000 + valid_window),valid_size // 2))
	num_sampled = 64

	graph_skipgram = tf.Graph()
	with graph_skipgram.as_default():
		train_dataset = tf.placeholder(tf.int32, shape = [batch_size])
		train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
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
		for step in range(num_steps):
			batch_data, batch_labels = generate_batch(
				batch_size = batch_size,
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
		skipgram_embeddings = normalized_embeddings.eval()
		saver = tf.train.Saver()
		saver.save(session, FLAGS.ckpt_path)
