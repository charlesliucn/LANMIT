#!/bin/bash
set -e
set -o pipefail

dir=data/fast_embed_concat
mkdir -p $dir

skip_size=100
skip_batch_size=128
skip_ckpt=$dir/skip_ckpt

cbow_size=100
cbow_batch_size=128
cbow_ckpt=$dir/cbow_ckpt

num_layers=1
batch_size=64
keep_prob=0.8
max_epoch=30

echo "$0 $@"
. ./utils/parse_options.sh

echo "====================================================================================="
echo "Usage of this scipt:"
echo "  $0 <the-lm>(language-model) <embed-method>(embedding-method)"
echo "The scipt provides three kinds of language model and three embedding methods as well"
echo "	Language Models: rnnlm lstmlm fast(lstm_fast) "
echo "	Embedding Methods: skipgram cbow glove "
echo "If you want to train the word embeddings during the neural network training: "
echo "  the embedding method can be set as base, which means it is the basic method."
echo "For example: "
echo "  $0 --the-lm rnnlm --embed-method skipgram"
echo "====================================================================================="

local/tfrnnlm/rnnlm_data_prep.sh $dir
mkdir -p $dir

echo "-----------------------------------"
echo " python steps/tfrnnlm/skip-gram.py "
echo "-----------------------------------"
# the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
python steps/tfrnnlm/skip-gram.py \
  --data-path=$dir --vocab-path=$dir/wordlist.rnn.final \
  --embedding-size=${skip_size} --batch-size=${skip_batch_size} \
  --ckpt-path=${skip_ckpt}

echo "-----------------------------------"
echo "    python steps/tfrnnlm/cbow.py   "
echo "-----------------------------------"
python steps/tfrnnlm/cbow.py \
  --data-path=$dir --vocab-path=$dir/wordlist.rnn.final \
  --embedding-size=${cbow_size} --batch-size=${cbow_batch_size} \
  --ckpt-path=${cbow_ckpt}

echo "-----------------------------------"
echo "python steps/tfrnnlm/fast_concat.py"
echo "-----------------------------------"

python steps/tfrnnlm/fast_concat.py \
  --data-path=$dir --vocab-path=$dir/wordlist.rnn.final \
  --skip-ckpt=${skip_ckpt} --cbow-ckpt=${cbow_ckpt} \
  --save-path=$dir/rnnlm --skip-size=${skip_size} \
  --cbow-size=${cbow_size} --num-layers=${num_layers} \
  --batch-size=${batch_size} --keep-prob=${keep_prob} \
  --max-epoch=${max_epoch}
