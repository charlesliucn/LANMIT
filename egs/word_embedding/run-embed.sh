#!/bin/bash

echo "rnnlm.skipgram"
./run_newlm.sh --the-lm rnnlm --embed-method skipgram \
	--hidden-size 200 --num-layers 2 --batch-size 128 \
	--keep-prob 0.5 --max-epoch 25 | tee logs/rnnlm.skip.200.1.128.0.5.log

echo "rnnlm.cbow"
./run_newlm.sh --the-lm rnnlm --embed-method cbow \
	--hidden-size 200 --num-layers 2 --batch-size 128 \
	--keep-prob 0.5 --max-epoch 25 | tee logs/rnnlm.cbow.200.2.128.0.5.log

echo "rnnlm.glove"
./run_newlm.sh --the-lm rnnlm --embed-method glove \
	--hidden-size 200 --num-layers 2 --batch-size 128 \
	--keep-prob 0.5 --max-epoch 25 | tee logs/rnnlm.glove.200.2.128.0.5.log

echo "lstm.skipgram"
./run_newlm.sh --the-lm lstm --embed-method skipgram \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.skip.200.1.64.0.8.log

echo "lstm.cbow"
./run_newlm.sh --the-lm lstm --embed-method cbow \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.cbow.200.1.64.0.8.log

echo "lstm.glove"
./run_newlm.sh --the-lm lstm --embed-method glove \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.glove.200.1.64.0.8.log

echo "fast.skipgram"
./run_newlm.sh --the-lm fast --embed-method skipgram \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.skip.200.1.64.0.8.log

echo "fast.cbow"
./run_newlm.sh --the-lm fast --embed-method cbow \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.cbow.200.1.64.0.8.log

echo "fast.glove"
./run_newlm.sh --the-lm fast --embed-method glove \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.glove.200.1.64.0.8.log
