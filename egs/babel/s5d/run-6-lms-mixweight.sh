#!/bin/bash

set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

add_countk=500
mix_lm=
data_dir=data_bnf

echo "$0 $@"

. utils/parse_options.sh

local/train_lms_mixweight.sh --main-lm ${data_dir}/srilm_${add_countk}k/lm.gz --mix-lm ${data_dir}/${mix_lm} \
	--main-ppl ${data_dir}/srilm_${add_countk}k/main.log --mix-ppl ${data_dir}/srilm_mixk/mix.log \
	--output-log mixlm.log --dev-text ${data_dir}/srilm_${add_countk}k/dev.txt
