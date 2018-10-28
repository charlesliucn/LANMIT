#!/bin/bash
set -e

dir=data/lstm_base
mkdir -p $dir
#steps/tfrnnlm/check_tensorflow_installed.sh
local/tfrnnlm/rnnlm_data_prep.sh $dir
mkdir -p $dir
# the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
python steps/tfrnnlm/lstm_base.py --data-path=$dir --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final
