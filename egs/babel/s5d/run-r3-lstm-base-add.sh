#!/bin/bash

data_dir=data_bnf
exp_dir=exp_bnf
tmp_dir=
ngram_order=4 # this option when used, the rescoring binary makes an approximation
    # to merge the states of the FST generated from RNNLM. e.g. if ngram-order = 4
    # then any history that shares last 3 words would be merged into one state
stage=1
weight=0.5   # when we do lattice-rescoring, instead of replacing the lm-weights
    # in the lattice with RNNLM weights, we usually do a linear combination of
    # the 2 and the $weight variable indicates the weight for the RNNLM scores

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

dir=${data_dir}/lstm_base
mkdir -p $dir

steps/tfrnnlm/check_tensorflow_installed.sh

if [ $stage -le 1 ]; then
  local/tfrnnlm/rnnlm_data_prep.sh $dir
fi

mkdir -p $dir
if [ $stage -le 2 ]; then
# the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
  $cuda_cmd $dir/train_lstm.log utils/parallel/limit_num_gpus.sh \
    python steps/tfrnnlm/lstm_base.py --data-path=$dir \
    --save-path=$dir/rnnlm --vocab-path=$dir/wordlist.rnn.final \
    --hidden-size=200 --num-layers=1 --batch-size=64 \
    --keep-prob=0.8 --max-epoch=25
fi

if [ $stage -le 3 ]; then
  decode_dir=${exp_dir}/sgmm5_mmi_b0.1/decode_fmllr_dev10h_bnf_it2
  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$tfrnnlm_cmd --mem 10G" \
    --weight $weight \
    --max-ngram-order $ngram_order \
    ${data_dir}/lang $dir \
    ${data_dir}/dev10h_bnf ${decode_dir} \
    ${decode_dir}.${tmp_dir}  &
fi
