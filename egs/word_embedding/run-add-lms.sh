#!/bin/bash

echo "add 100k 300"
[ -d data/fast_base ] && rm -rf data/fast_base/
./run_newlm.sh --the-lm fast --embed-method base \
	--hidden-size 300 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.100k.300.1.64.0.8.log

echo "add 100k 400"
[ -d data/fast_base ] && rm -rf data/fast_base/
./run_newlm.sh --the-lm fast --embed-method base \
	--hidden-size 400 --num-layers 1 --batch-size 64 \
	--keep-prob 0.8 --max-epoch 25 | tee logs/fast.100k.400.1.64.0.8.log
