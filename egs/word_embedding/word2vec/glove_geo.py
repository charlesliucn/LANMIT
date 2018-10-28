# -*- coding:utf-8 -*-

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from scipy.sparse import lil_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.Session(config = config)

def read_data(filename):
	with open(filename, "r") as f:
		texts = f.read()
		data = texts.split()
		f.close()
	return data

filename = "train_text"
words = read_data(filename)
print('Data size %d' % len(words))
print('Sample string %s' % words[:50])

vocabulary_size = 30000

def build_dataset(words):
	count = [["UNK", -1]]
	wordscounts = collections.Counter(words)
	words_common = wordscounts.most_common(vocabulary_size - 1)
	count.extend(words_common)
	
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
			unk_count = unk_count + 1
		data.append(index)
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
	return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
	global data_index
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window
	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	weights = np.ndarray(shape=(batch_size), dtype=np.float32)
	span = 2 * skip_window + 1
	buffer = collections.deque(maxlen=span)
	for _ in range(span):
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	for i in range(batch_size // num_skips):
		target = skip_window
		targets_to_avoid = [ skip_window ]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buffer[skip_window]
			labels[i * num_skips + j, 0] = buffer[target]
			weights[i * num_skips + j] = abs(1.0/(target - skip_window))
		buffer.append(data[data_index])
		data_index = (data_index + 1) % len(data)
	return batch, labels, weights

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (8, 4)]:
	data_index = 0
	batch, labels, weights = generate_batch(
		batch_size = 8,
		num_skips = num_skips, 
		skip_window = skip_window)
	print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
	print('    batch:', [reverse_dictionary[bi] for bi in batch])
	print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])
	print('    weights:', [w for w in weights])

cooc_data_index = 0
dataset_size = len(data)
skip_window = 4
num_skips = 8

cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)
print(cooc_mat.shape)
def generate_cooc(batch_size,num_skips,skip_window):
	data_index = 0
	print('Running %d iterations to compute the co-occurance matrix' %( dataset_size // batch_size))
	for i in range(dataset_size//batch_size):
		if i > 0 and i % 100000 == 0:
			print('\tFinished %d iterations' % i)
		batch, labels, weights = generate_batch(
			batch_size = batch_size,
			num_skips = num_skips,
			skip_window = skip_window) # increments data_index automatically
		labels = labels.reshape(-1)
		for inp,lbl,w in zip(batch,labels,weights):            
			cooc_mat[inp,lbl] += (1.0*w)
						
generate_cooc(8,num_skips,skip_window)    

print('Sample chunks of co-occurance matrix')
rand_target_idx = np.random.randint(0,vocabulary_size,10).tolist()
for i in range(10):
	idx_target = i
	ith_row = cooc_mat.getrow(idx_target)

	ith_row_dense = ith_row.toarray('C').reshape(-1)

	while np.sum(ith_row_dense) < 10 or np.sum(ith_row_dense)>50000:
		idx_target = np.random.randint(0,vocabulary_size)
		ith_row = cooc_mat.getrow(idx_target)
		ith_row_dense = ith_row.toarray('C').reshape(-1)    
				
	print('\nTarget Word: "%s"' % reverse_dictionary[idx_target])
			
	sort_indices = np.argsort(ith_row_dense).reshape(-1) # indices with highest count of ith_row_dense
	sort_indices = np.flip(sort_indices,axis=0) # reverse the array (to get max values to the start)

	# printing several context words to make sure cooc_mat is correct
	print('Context word:',end='')
	for j in range(10):        
		idx_context = sort_indices[j]       
		print('"%s"(id:%d,count:%.2f), '%(reverse_dictionary[idx_context],idx_context,ith_row_dense[idx_context]),end='')
	print()

if __name__ == '__main__':
	batch_size = 128
	embedding_size = 192 # Dimension of the embedding vector.

	# We pick a random validation set to sample nearest neighbors. here we limit the
	# validation samples to the words that have a low numeric ID, which by
	# construction are also the most frequent. 
	valid_size = 16 # Random set of words to evaluate similarity on.
	valid_window = 100 # Only pick dev samples in the head of the distribution.

	# Validation set consist of 50 infrequent words and 50 frequent words
	valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
	valid_examples = np.append(valid_examples,random.sample(range(1000,1000+valid_window), valid_size//2))

	epsilon = 1 # used for the stability of log in the loss function
	graph = tf.Graph()

	with graph.as_default(), tf.device('/cpu:0'):

		# Input data.
		train_dataset = tf.placeholder(tf.int32, shape=[batch_size],name='train_dataset')
		train_labels = tf.placeholder(tf.int32, shape=[batch_size],name='train_labels')
		valid_dataset = tf.constant(valid_examples, dtype=tf.int32,name='valid_dataset')

		# Variables.
		embeddings = tf.Variable(
				tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),name='embeddings')
		bias_embeddings = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01,dtype=tf.float32),name='embeddings_bias')

		# Model.
		# Look up embeddings for inputs.
		embed_in = tf.nn.embedding_lookup(embeddings, train_dataset)
		embed_out = tf.nn.embedding_lookup(embeddings, train_labels)
		embed_bias_in = tf.nn.embedding_lookup(bias_embeddings,train_dataset)
		embed_bias_out = tf.nn.embedding_lookup(bias_embeddings,train_labels)

		# weights used in the cost function
		weights_x = tf.placeholder(tf.float32,shape=[batch_size],name='weights_x') 
		x_ij = tf.placeholder(tf.float32,shape=[batch_size],name='x_ij')

		# Compute the loss defined in the paper. Note that I'm not following the exact equation given (which is computing a pair of words at a time)
		# I'm calculating the loss for a batch at one time, but the calculations are identical.
		# I also made an assumption about the bias, that it is a smaller type of embedding
		loss = tf.reduce_mean(
				weights_x * (tf.reduce_sum(embed_in*embed_out, axis=1) + embed_bias_in + embed_bias_out - tf.log(epsilon+x_ij))**2)

		# Optimizer.
		optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

		# Compute the similarity between minibatch examples and all embeddings.
		norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
		normalized_embeddings = embeddings / norm
		valid_embeddings = tf.nn.embedding_lookup(
		normalized_embeddings, valid_dataset)
		similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


		num_steps = 100001
		session = tf.InteractiveSession()
		tf.global_variables_initializer().run()
		print('Initialized')
		average_loss = 0
		for step in range(num_steps):
			batch_data, batch_labels, batch_weights = generate_batch(
				batch_size, num_skips, skip_window) # generate a single batch (data,labels,co-occurance weights)
			batch_weights = [] # weighting used in the loss function
			batch_xij = [] # weighted frequency of finding i near j
			for inp,lbl in zip(batch_data,batch_labels.reshape(-1)):        
				batch_weights.append((np.asscalar(cooc_mat[inp,lbl])/100.0)**0.75)
				batch_xij.append(cooc_mat[inp,lbl])
			batch_weights = np.clip(batch_weights,-100,1)
			batch_xij = np.asarray(batch_xij)

			feed_dict = {train_dataset : batch_data.reshape(-1), train_labels : batch_labels.reshape(-1),
				weights_x:batch_weights,x_ij:batch_xij}
			_, l = session.run([optimizer, loss], feed_dict=feed_dict)

			average_loss += l
			if step % 2000 == 0:
				if step > 0:
					average_loss = average_loss / 2000
				print('Average loss at step %d: %f' % (step, average_loss))
				average_loss = 0
			# note that this is expensive (~20% slowdown if computed every 500 steps)
			if step % 10000 == 0:
				sim = similarity.eval()
				for i in range(valid_size):
					valid_word = reverse_dictionary[valid_examples[i]]
					top_k = 8 # number of nearest neighbors
					nearest = (-sim[i, :]).argsort()[1:top_k+1]
					log = 'Nearest to %s:' % valid_word
					for k in range(top_k):
						close_word = reverse_dictionary[nearest[k]]
						log = '%s %s,' % (log, close_word)
					print(log)
		final_embeddings = normalized_embeddings.eval()