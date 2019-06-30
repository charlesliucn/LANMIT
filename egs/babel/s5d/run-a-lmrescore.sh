#!/bin/bash

# This is not necessarily the top-level run.sh as it is in other directories.
# see README.txt first.
data_dir=data_bnf
exp_dir=exp_bnf

[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
[ ! -f ./conf/common_vars.sh ] && echo 'the file conf/common_vars.sh does not exist!' && exit 1

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

[ -f local.conf ] && . ./local.conf

. ./utils/parse_options.sh

set -e           #Exit on non-zero return code from any command
set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
#set -u           #Fail on an undefined variable

echo "--------------------------------------------------------------------"
echo "Do language model rescoring of lattices (remove old LM, add new LM)"
echo "Usage: steps/lmrescore.sh [options] <old-lang-dir> <new-lang-dir> <data-dir> <input-decode-dir> <output-decode-dir>"
echo "--------------------------------------------------------------------"

steps/lmrescore.sh --cmd "$decode_cmd" \
 ${data_dir}/lang_500k \
 ${data_dir}/lang_mixk \
 ${data_dir}/dev10h_bnf \
 ${exp_dir}/sgmm5_mmi_b0.1/decode_fmllr_dev10h_bnf_it2 \
 ${exp_dir}/sgmm5_mmi_b0.1/decode_fmllr_dev10h_bnf_it2_re3.2

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
