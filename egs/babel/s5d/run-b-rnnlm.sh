#!/bin/bash

. ./cmd.sh
. ./path.sh

data_dir=data_bnf
exp_dir=exp_bnf
# This script demonstrates how you can train rnnlms, and how you can use them to
# rescore the n-best lists, or lattices.
# Be careful: appending things like "--mem 16G" to $decode_cmd won't always
# work, it depends what $decode_cmd is.

# Trains Tomas Mikolov's version, which takes roughly 5 days with the following
# parameter setting. We start from the dictionary directory without silence
# probabilities (with suffix "_nosp").
# rm data/local/rnnlm.h300.voc40k/.error 2>/dev/null
# local/wsj_train_rnnlms.sh --dict-suffix "_nosp" \
#   --cmd "$decode_cmd --mem 16G" \
#   --hidden 300 --nwords 40000 --class 400 \
#   --direct 2000 data/local/rnnlm.h300.voc40k \
#   || touch data/local/rnnlm.h300.voc40k/.error &

# Trains Yandex's version, which takes roughly 10 hours with the following
# parameter setting. We start from the dictionary directory without silence
# probabilities (with suffix "_nosp").
num_threads_rnnlm=2

# rm ${data_dir}/rnnlm_dir/rnnlm-faster.nce20.h200.voc30k/.error 2>/dev/null
local/train_rnnlms.sh \
  --rnnlm_ver faster-rnnlm --threads $num_threads_rnnlm \
  --cmd "$decode_cmd --mem 2G --num-threads $num_threads_rnnlm" \
  --bptt 4 --bptt-block 20 --hidden 200 --nwords 30000 --direct 1500 \
  --rnnlm-options "-direct-order 4 -nce 20" \
  ${data_dir}/rnnlm_dir/rnnlm-faster.nce20.h200.voc30k \
  || touch ${data_dir}/rnnlm_dir/rnnlm-faster.nce20.h200.voc30k/.error &

wait;

# Rescoring. We demonstrate results on the TDNN models. Make sure you have
# finished running the following scripts:
#   local/online/run_nnet2.sh
#   local/online/run_nnet2_baseline.sh
#   local/online/run_nnet2_discriminative.sh

# # N-best rescoring with Tomas Mikolov's version.
# steps/rnnlmrescore.sh \
#   --N 1000 --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.75 \
#   data/lang_test_${lm_suffix} data/local/rnnlm.h300.voc40k \
#   data/test_${year} ${decode_dir} \
#   ${decode_dir}_rnnlm.h300.voc40k || exit 1;

# # Lattice rescoring with Tomas Mikolov's version.
# steps/lmrescore_rnnlm_lat.sh \
#   --weight 0.75 --cmd "$decode_cmd --mem 16G" --max-ngram-order 5 \
#   data/lang_test_${lm_suffix} data/local/rnnlm.h300.voc40k \
#   data/test_${year} ${decode_dir} \
#   ${decode_dir}_rnnlm.h300.voc40k_lat || exit 1;

# # N-best rescoring with Yandex's version.
# decode_dir=${exp_dir}/sgmm5_mmi_b0.1/decode_fmllr_dev10h_bnf_it2
# steps/rnnlmrescore.sh --rnnlm_ver faster-rnnlm \
#   --N 1000 --cmd "$decode_cmd --mem 3G" --inv-acwt 10 0.75 \
#   ${data_dir}/lang_mixk ${data_dir}/rnnlm_dir/rnnlm-faster.nce20.h200.voc30k \
#   ${data_dir}/dev10h_bnf ${decode_dir} \
#   ${decode_dir}_rnnlm_1 || exit 1;
