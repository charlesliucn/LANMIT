# -*- coding:utf-8 -*-

from gensim import models
import os, sys
import numpy as np
import collections

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
	inferred_vector = model.infer_vector(test_doc)
	sims_test = model.docvecs.most_similar(
		[inferred_vector],
		topn = len(model.docvecs))
	#---------------------------------------------
	# sanity check: check for self-similarity
	print("---------------------------------")
	print("        Doing sanity check       ")
	print("---------------------------------")
	ranks = []
	for doc_id in range(len(train_corpus)):
		inferred_vector = model.infer_vector(train_corpus[doc_id].words)
		sims = model.docvecs.most_similar(
			[inferred_vector],
			topn = len(model.docvecs))
		rank = [doci for doci, sim in sims].index(doc_id)
		ranks.append(rank)
	return sims_test, ranks

def main():
	sims_test, ranks = train_model(60500, 300, 80)
	# sims_test, ranks = train_model(1000, 100, 100)
	# print("-------------")
	# print(sims_test)
	print("-------------")
	print(len(sims_test))
	print("-------------")
	# print(ranks)
	print(collections.Counter(ranks))
	print("-------------")
	os.system("[ -f doc2vec.txt ] && rm doc2vec.txt")
	rfile = open("doc2vec.txt", "a")
	order = []
	for i, score in sims_test:
		order.append(i+1)
		docname = "./docs/"+str(i+1)+".txt"
		tmpfile = open(docname, "r")
		tmptext = tmpfile.read()
		tmpfile.close()
		rfile.write(tmptext)
	rfile.close()
	# print(order)

if __name__ == '__main__':
	main()