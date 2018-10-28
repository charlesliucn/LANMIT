#!/bin/bash


# [ -d data/rnnlm_skipgram ] && rm -rf data/rnnlm_skipgram
# ./run_newlm.sh --the-lm rnnlm --embed-method skipgram \
# 	--hidden-size 200 --num-layers 2 --batch-size 128 \
# 	--keep-prob 0.5 --max-epoch 25 | tee logs/rnn.skip.200.2.128.0.5.log

# [ -d data/lstm_skipgram ] && rm -rf data/lstm_skipgram
# ./run_newlm.sh --the-lm lstm --embed-method skipgram \
# 	--hidden-size 100 --num-layers 1 --batch-size 64 \
# 	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.skip.100.1.64.0.8.log

[ -d data/fast_skipgram ] && rm -rf data/fast_skipgram
./run_newlm.sh --the-lm fast --embed-method skipgram \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.skip.200.1.64.0.8.log

# [ -d data/lstm_cbow ] && rm -rf data/lstm_cbow
# ./run_newlm.sh --the-lm lstm --embed-method cbow \
# 	--hidden-size 100 --num-layers 1 --batch-size 64 \
# 	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.cbow.100.1.64.0.8.log

[ -d data/lstm_cbow ] && rm -rf data/lstm_cbow
./run_newlm.sh --the-lm lstm --embed-method cbow \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.cbow.200.1.64.0.8.log

[ -d data/fast_cbow ] && rm -rf data/fast_cbow
./run_newlm.sh --the-lm fast --embed-method cbow \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.cbow.200.1.64.0.8.log

[ -d data/lstm_glove ] && rm -rf data/lstm_glove
./run_newlm.sh --the-lm lstm --embed-method glove \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/lstm.glove.200.1.64.0.8.log

[ -d data/fast_glove ] && rm -rf data/fast_glove
./run_newlm.sh --the-lm fast --embed-method glove \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.glove.200.1.64.0.8.log