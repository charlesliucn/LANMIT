#!/bin/bash
set -e
set -o pipefail

the_lm=rnnlm
embed_method=base
hidden_size=200
num_layers=2
batch_size=64
keep_prob=1.0
max_epoch=25

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

dir=data/${the_lm}_${embed_method}
mkdir -p $dir

local/tfrnnlm/rnnlm_data_prep.sh $dir
mkdir -p $dir

# the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
python steps/tfrnnlm/${the_lm}_${embed_method}.py \
  --data-path=$dir --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final \
  --hidden-size=${hidden_size} --num-layers=${num_layers} --batch-size=${batch_size} \
  --keep-prob=${keep_prob} --max-epoch=${max_epoch}