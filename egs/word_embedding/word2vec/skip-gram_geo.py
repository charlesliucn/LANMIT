# -*- coding:utf-8 -*-

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config = config)


def read_data(filename):
	with open(filename) as f:
		g = f.read()
		data = g.split()
		f.close()
	return data

filename = "train_text"
words = read_data(filename)
print("Data Size(Total number of words): %d" % len(words))

vocabulary_size = 30000

def build_dataset(words):
	count = [["UNK", -1]]
	wordscounts = collections.Counter(words)
	words_common = wordscounts.most_common(vocabulary_size - 1)
	count.extend(words_common)
	# print(count)
	dictionary = dict()
	for word, _ in count:
		dictionary[word] = len(dictionary)
	data = list()
	unk_count = 0
	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0
			unk_count += 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print("Most 10 common words (including UNK):", count[:10])
print("Sample data (the first 10 words(id)):", data[:10])
del words


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
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)

	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels

print("data:", [reverse_dictionary[idx] for idx in data[:8]])
# Display parts of the batch and labels
for num_skips, skip_window in [(2, 1), (4, 2)]:
	data_index = 0
	batch, labels = generate_batch(
		batch_size = 8,
		num_skips = num_skips,
		skip_window = skip_window)

	print("\n with num_skips=%d and skip_window=%d" % (num_skips, skip_window))
	print("  batch: ", [reverse_dictionary[idx] for idx in batch])
	print("  labels: ", [reverse_dictionary[idx] for idx in labels.reshape(8)])


num_steps = 100001
if __name__ == '__main__':
	batch_size = 128
	embedding_size = 128
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
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8
					nearest = (-sim[i,:]).argsort()[1:top_k+1]
					log = "Nearest to %s:" % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log = log + " " + close_word + ","
					print(log)
		final_embeddings = normalized_embeddings.eval()

	