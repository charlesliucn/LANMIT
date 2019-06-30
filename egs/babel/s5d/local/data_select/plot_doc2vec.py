# -*- coding:utf-8 -*-

from gensim import models
import os, sys
import numpy as np
import pandas as pd
import collections
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

def docs_reader(directory_name, file_num):
	# read in all the documents as training corpus
	print("---------------------------------")
	print("Read all the documents as corpus")
	print("---------------------------------")
	texts = []
	for i in range(file_num):
		filename = directory_name+"/"+str(i+1)+".txt"
		f = open(filename)
		doc = f.read()
		f.close()
		new_text = [word for word in doc.split()]
		# add document tags as Tagged Document
		texts.append(models.doc2vec.TaggedDocument(new_text, [i]))
	# print(texts)
	print("---------------------------------")
	print("           Done Reading          ")
	print("---------------------------------")
	return texts

def train_model(file_num, vectorsize, epochs_num):
	#---------------------------------------------
	docs_dir = "./docs"
	#---------------------------------------------
	# Obtain train_corpus
	train_corpus = docs_reader(docs_dir, file_num)
	# Instantiate a Doc2Vec Model Object
	model = models.doc2vec.Doc2Vec(
		vector_size = vectorsize,
		min_count = 2,
		epochs = epochs_num)
	#---------------------------------------------
	print("---------------------------------")
	print("          Building Vocab         ")
	print("---------------------------------")
	# Build a Vocabulary
	model.build_vocab(train_corpus)
	#---------------------------------------------
	print("---------------------------------")
	print("       Training the model        ")
	print("---------------------------------")
	# Train the model
	model.train(
		train_corpus,
		total_examples = model.corpus_count, 
		epochs = model.epochs)
	#---------------------------------------------
	# Testing the model (Read the Test Document)
	ref_docname = "./dtrain.txt"
	f = open(ref_docname, "r")
	doc = f.read()
	f.close()
	test_doc = [word for word in doc.split()]
	#---------------------------------------------
	# Obtain the inferred vector based on the model
	inferred_vector_test = model.infer_vector(test_doc)
	#---------------------------------------------
	# sanity check: check for self-similarity
	print("---------------------------------")
	print("        Doing sanity check       ")
	print("---------------------------------")
	ranks = []
	inferred_vectors = []
	for doc_id in range(len(train_corpus)):
		inferred_vector = model.infer_vector(train_corpus[doc_id].words)
		inferred_vectors.append(inferred_vector)
	inferred_vectors.append(inferred_vector_test)
	return inferred_vectors

def main():
	ndoc = 60500
	inferred_vectors = train_model(ndoc, 200, 80)
	infvec_dataframe = DataFrame(inferred_vectors)
	print(infvec_dataframe)
	print(len(infvec_dataframe))
	X_tsne = TSNE(n_components = 2,learning_rate = 100).fit_transform(infvec_dataframe)
	plt.figure(figsize = (12, 6))
	col = ["blue"]*ndoc
	col.append("red")
	plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c = col, marker = ".")
	plt.xlabel("(Dimension 1)")
	plt.ylabel("(Dimension 2)")
	plt.title("t-SNE Method to Visualize the Documents")
	plt.show()

if __name__ == '__main__':
	main()