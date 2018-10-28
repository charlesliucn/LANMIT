#!/bin/bash
set -e
set -o pipefail

. conf/common_vars.sh || exit 1;
. ./lang.conf || exit 1;

data_dir=data_bnf
exp_dir=exp_bnf
dir=dev10h_bnf
dataset_id=$dir
dataset_dir=${data_dir}/$dir
dataset_type=dev10h
eval my_nj=\$${dataset_type}_nj


kind=
skip_kws=false
skip_stt=false
skip_scoring=false
extra_kws=true
vocab_kws=false
tri5_only=false
wip=0.5

nnet3_model=nnet3/tdnn_sp
chain_model=
parent_dir_suffix=_cleaned
is_rnn=false
extra_left_context=40
extra_right_context=40
frames_per_chunk=20

add_countk=
echo "$0 $@"

. utils/parse_options.sh

if [ $# -ne 0 ]; then
  echo "Usage: $(basename $0) --type (dev10h|dev2h|eval|shadow)"
  exit 1
fi

echo "Dir: $dir"

set -o errtrace
trap "echo Exited!; exit;" SIGINT SIGTERM

./local/check_tools.sh || exit 1

####################################################################
##
## FMLLR decoding
##
####################################################################
decode=${exp_dir}/tri5/decode_${dataset_id}
if [ ! -f ${decode}/.done ]; then
  echo ---------------------------------------------------------------------
  echo "Spawning decoding with SAT models  on" `date`
  echo ---------------------------------------------------------------------
  utils/mkgraph.sh \
    ${data_dir}/lang ${exp_dir}/tri5 ${exp_dir}/tri5/graph | tee ${exp_dir}/tri5/mkgraph.log

  mkdir -p $decode
  #By default, we do not care about the lattices for this step -- we just want the transforms
  #Therefore, we will reduce the beam sizes, to reduce the decoding times
  steps/decode_fmllr_extra.sh --skip-scoring false --beam 10 --lattice-beam 4\
    --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
    ${exp_dir}/tri5/graph ${dataset_dir} ${decode} | tee ${decode}/decode.log
  touch ${decode}/.done
fi

####################################################################
## SGMM2 decoding
## We Include the SGMM_MMI inside this, as we might only have the DNN systems
## trained and not PLP system. The DNN systems build only on the top of tri5 stage
####################################################################
if [ -f ${exp_dir}/sgmm5/.done ]; then
  decode=${exp_dir}/sgmm5/decode_fmllr_${dataset_id}
  if [ ! -f $decode/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Spawning $decode on" `date`
    echo ---------------------------------------------------------------------
    utils/mkgraph.sh \
      ${data_dir}/lang ${exp_dir}/sgmm5 ${exp_dir}/sgmm5/graph | tee ${exp_dir}/sgmm5/mkgraph.log

    mkdir -p $decode
    steps/decode_sgmm2.sh --skip-scoring false --use-fmllr true --nj $my_nj \
      --cmd "$decode_cmd" --transform-dir ${exp_dir}/tri5/decode_${dataset_id} "${decode_extra_opts[@]}"\
      ${exp_dir}/sgmm5/graph ${dataset_dir} $decode | tee $decode/decode.log
    touch $decode/.done

  fi

  ####################################################################
  ##
  ## SGMM_MMI rescoring
  ##
  ####################################################################

  for iter in 1 2 3 4; do
      # Decode SGMM+MMI (via rescoring).
    decode=${exp_dir}/sgmm5_mmi_b0.1/decode_fmllr_${dataset_id}_it$iter
    if [ -x ${exp_dir}/sgmm5_mmi_b0.1 ] && [ ! -f $decode/.done ]; then

      mkdir -p $decode
      steps/decode_sgmm2_rescore.sh  --skip-scoring false \
        --cmd "$decode_cmd" --iter $iter --transform-dir ${exp_dir}/tri5/decode_${dataset_id} \
        ${data_dir}/lang ${dataset_dir} ${exp_dir}/sgmm5/decode_fmllr_${dataset_id} $decode | tee ${decode}/decode.log

      touch $decode/.done
    fi
  done

fi

echo "Done SGMM scoring and SGMM_MMI rescoring!"
echo "Please run script run-results.sh to obtain the WER results!"
exit 0
