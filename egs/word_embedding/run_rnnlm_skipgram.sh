#!/bin/bash
set -e

dir=data/rnn_skipgram # 设置RNNLM的输入/输出文件夹
[ -d $dir ] && rm -rf $dir
mkdir -p $dir

# steps/tfrnnlm/check_tensorflow_installed.sh
local/tfrnnlm/rnnlm_data_prep.sh $dir # 准备RNNLM的数据

mkdir -p $dir
# 调用python脚本文件vanilla_rnnlm.py
# 数据路径：data/vanilla_tensorflow
# 存储路径：data/vanilla_tensorflow
# 词汇路径：data/vanilla_tensorflow/wordlist.rnn.final

echo "-----------------------------------"
echo "      RNNLM Training Using GPU!    "
echo "-----------------------------------"

python steps/tfrnnlm/rnnlm_skipgram.py --data-path=$dir --save-path=$dir --vocab-path=$dir/wordlist.rnn.final
