#!/bin/bash

echo "----------------------------------------------------"
echo "Training RNNLM:"
echo "./run_newlm.sh --the-lm rnnlm --embed-method skipgram \
	 --hidden-size 200 --num-layers 1 --batch-size 64"
echo "----------------------------------------------------"
[ -d data/fast_skipgram ] && rm -rf data/fast_skipgram
./run_newlm.sh --the-lm fast --embed-method skipgram \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.skip.200.1.64.0.8.log

[ -d data/fast_cbow ] && rm -rf data/fast_cbow
./run_newlm.sh --the-lm fast --embed-method cbow \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.cbow.200.1.64.0.8.log

[ -d data/fast_glove ] && rm -rf data/fast_glove
./run_newlm.sh --the-lm fast --embed-method glove \
	--hidden-size 200 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.glove.200.1.64.0.8.log
