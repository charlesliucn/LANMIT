#!/bin/bash

set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

main_lambda=
mix_lm=
out_mixlm=
add_countk=500
data_dir=data_bnf

echo "$0 $@"

. utils/parse_options.sh

local/train_merge_lms.sh --main-lm ${data_dir}/srilm_${add_countk}k/lm.gz  \
  --mix-lm ${data_dir}/${mix_lm} --main-lambda ${main_lambda} \
  --merge-lm ${data_dir}/srilm_mixk/${out_mixlm}
