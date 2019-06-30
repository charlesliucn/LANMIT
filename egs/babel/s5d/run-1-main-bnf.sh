#!/bin/bash

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.
tag_percentage=0.1
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

lexicon=${data_dir}/local/lexicon.txt

./local/check_tools.sh || exit 1

#Preparing dev10h and train directories
if [ ! -f ${data_dir}/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./${data_dir}/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./${data_dir}/raw_train_data`
    touch ${data_dir}/raw_train_data/.done
fi
nj_max=`cat $train_data_list | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`utils/make_absolute.sh ./${data_dir}/raw_train_data`

if [ ! -d ${data_dir}/raw_dev10h_data ]; then
  echo ---------------------------------------------------------------------
  echo "Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------
  local/make_corpus_subset.sh "$dev10h_data_dir" "$dev10h_data_list" ./${data_dir}/raw_dev10h_data || exit 1
fi

mkdir -p ${data_dir}/lang
if [[ ! -f ${data_dir}/lang/L.fst || ${data_dir}/lang/L.fst -ot $lexicon ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in ${data_dir}/lang on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh \
    --share-silence-phones true \
    ${data_dir}/local $oovSymbol ${data_dir}/local/tmp.lang ${data_dir}/lang
fi

if [[ ! -f ${data_dir}/srilm/lm.gz || ${data_dir}/srilm/lm.gz -ot ${data_dir}/train/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM language models on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm.sh  --oov-symbol "$oovSymbol"\
    --words-file ${data_dir}/lang/words.txt \
    --train-text ${data_dir}/train/text \
    --dev-text ${data_dir}/dev10h_bnf/text ${data_dir} ${data_dir}/srilm
fi

if [[ ! -f ${data_dir}/lang/G.fst || ${data_dir}/lang/G.fst -ot ${data_dir}/srilm/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh ${data_dir}/srilm/lm.gz ${data_dir}/lang ${data_dir}/lang
fi

mkdir -p ${exp_dir}

steps/align_fmllr.sh \
  --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
  ${data_dir}/dev10h_bnf ${data_dir}/langp/tri5 ${exp_dir}/tri5 ${exp_dir}/tri5_dev10h_ali

# if [ ! -f ${data_dir}/train_sub3/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Subsetting monophone training data in ${data_dir}/train_sub[123] on" `date`
#   echo ---------------------------------------------------------------------
#   numutt=`cat ${data_dir}/train/feats.scp | wc -l`;
#   utils/subset_data_dir.sh ${data_dir}/train  5000 ${data_dir}/train_sub1
#   if [ $numutt -gt 10000 ] ; then
#     utils/subset_data_dir.sh ${data_dir}/train 10000 ${data_dir}/train_sub2
#   else
#     (cd ${data_dir}; ln -s train train_sub2 )
#   fi
#   if [ $numutt -gt 20000 ] ; then
#     utils/subset_data_dir.sh ${data_dir}/train 20000 ${data_dir}/train_sub3
#   else
#     (cd ${data_dir}; ln -s train train_sub3 )
#   fi

#   touch ${data_dir}/train_sub3/.done
# fi

# if [ ! -f ${exp_dir}/mono/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting (small) monophone training in ${exp_dir}/mono on" `date`
#   echo ---------------------------------------------------------------------
#   steps/train_mono.sh \
#     --boost-silence $boost_sil --nj 8 --cmd "$train_cmd" \
#     ${data_dir}/train_sub1 ${data_dir}/lang ${exp_dir}/mono
#   touch ${exp_dir}/mono/.done
# fi

# if [ ! -f ${exp_dir}/tri1/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting (small) triphone training in ${exp_dir}/tri1 on" `date`
#   echo ---------------------------------------------------------------------
#   steps/align_si.sh \
#     --boost-silence $boost_sil --nj 12 --cmd "$train_cmd" \
#     ${data_dir}/train_sub2 ${data_dir}/lang ${exp_dir}/mono ${exp_dir}/mono_ali_sub2

#   steps/train_deltas.sh \
#     --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri1 $numGaussTri1 \
#     ${data_dir}/train_sub2 ${data_dir}/lang ${exp_dir}/mono_ali_sub2 ${exp_dir}/tri1

#   touch ${exp_dir}/tri1/.done
# fi


# echo ---------------------------------------------------------------------
# echo "Starting (medium) triphone training in ${exp_dir}/tri2 on" `date`
# echo ---------------------------------------------------------------------
# if [ ! -f ${exp_dir}/tri2/.done ]; then
#   steps/align_si.sh \
#     --boost-silence $boost_sil --nj 24 --cmd "$train_cmd" \
#     ${data_dir}/train_sub3 ${data_dir}/lang ${exp_dir}/tri1 ${exp_dir}/tri1_ali_sub3

#   steps/train_deltas.sh \
#     --boost-silence $boost_sil --cmd "$train_cmd" $numLeavesTri2 $numGaussTri2 \
#     ${data_dir}/train_sub3 ${data_dir}/lang ${exp_dir}/tri1_ali_sub3 ${exp_dir}/tri2

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train_sub3 ${data_dir}/lang ${data_dir}/local/ \
#     ${exp_dir}/tri2 ${data_dir}/local/dictp/tri2 ${data_dir}/local/langp/tri2 ${data_dir}/langp/tri2

#   touch ${exp_dir}/tri2/.done
# fi

# echo ---------------------------------------------------------------------
# echo "Starting (full) triphone training in ${exp_dir}/tri3 on" `date`
# echo ---------------------------------------------------------------------
# if [ ! -f ${exp_dir}/tri3/.done ]; then
#   steps/align_si.sh \
#     --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
#     ${data_dir}/train ${data_dir}/langp/tri2 ${exp_dir}/tri2 ${exp_dir}/tri2_ali

#   steps/train_deltas.sh \
#     --boost-silence $boost_sil --cmd "$train_cmd" \
#     $numLeavesTri3 $numGaussTri3 ${data_dir}/train ${data_dir}/langp/tri2 ${exp_dir}/tri2_ali ${exp_dir}/tri3

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train ${data_dir}/lang ${data_dir}/local/ \
#     ${exp_dir}/tri3 ${data_dir}/local/dictp/tri3 ${data_dir}/local/langp/tri3 ${data_dir}/langp/tri3

#   touch ${exp_dir}/tri3/.done
# fi

# echo ---------------------------------------------------------------------
# echo "Starting (lda_mllt) triphone training in ${exp_dir}/tri4 on" `date`
# echo ---------------------------------------------------------------------
# if [ ! -f ${exp_dir}/tri4/.done ]; then
#   steps/align_si.sh \
#     --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
#     ${data_dir}/train ${data_dir}/langp/tri3 ${exp_dir}/tri3 ${exp_dir}/tri3_ali

#   steps/train_lda_mllt.sh \
#     --boost-silence $boost_sil --cmd "$train_cmd" \
#     $numLeavesMLLT $numGaussMLLT ${data_dir}/train ${data_dir}/langp/tri3 ${exp_dir}/tri3_ali ${exp_dir}/tri4

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train ${data_dir}/lang ${data_dir}/local \
#     ${exp_dir}/tri4 ${data_dir}/local/dictp/tri4 ${data_dir}/local/langp/tri4 ${data_dir}/langp/tri4

#   touch ${exp_dir}/tri4/.done
# fi

# echo ---------------------------------------------------------------------
# echo "Starting (SAT) triphone training in ${exp_dir}/tri5 on" `date`
# echo ---------------------------------------------------------------------

# if [ ! -f ${exp_dir}/tri5/.done ]; then
#   steps/align_si.sh \
#     --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
#     ${data_dir}/train ${data_dir}/langp/tri4 ${exp_dir}/tri4 ${exp_dir}/tri4_ali

#   steps/train_sat.sh \
#     --boost-silence $boost_sil --cmd "$train_cmd" \
#     $numLeavesSAT $numGaussSAT ${data_dir}/train ${data_dir}/langp/tri4 ${exp_dir}/tri4_ali ${exp_dir}/tri5

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train ${data_dir}/lang ${data_dir}/local \
#     ${exp_dir}/tri5 ${data_dir}/local/dictp/tri5 ${data_dir}/local/langp/tri5 ${data_dir}/langp/tri5

#   touch ${exp_dir}/tri5/.done
# fi


# if [ ! -f ${exp_dir}/tri5_ali/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/tri5_ali on" `date`
#   echo ---------------------------------------------------------------------
#   steps/align_fmllr.sh \
#     --boost-silence $boost_sil --nj $train_nj --cmd "$train_cmd" \
#     ${data_dir}/train ${data_dir}/langp/tri5 ${exp_dir}/tri5 ${exp_dir}/tri5_ali

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train ${data_dir}/lang ${data_dir}/local \
#     ${exp_dir}/tri5_ali ${data_dir}/local/dictp/tri5_ali ${data_dir}/local/langp/tri5_ali ${data_dir}/langp/tri5_ali

#   touch ${exp_dir}/tri5_ali/.done
# fi

# if [ ! -f ${data_dir}/langp_test/.done ]; then
#   cp -R ${data_dir}/langp/tri5_ali/ ${data_dir}/langp_test
#   cp ${data_dir}/lang/G.fst ${data_dir}/langp_test
#   touch ${data_dir}/langp_test/.done
# fi

# ################################################################################
# # Ready to start SGMM training
# ################################################################################

# if [ ! -f ${exp_dir}/ubm5/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/ubm5 on" `date`
#   echo ---------------------------------------------------------------------
#   steps/train_ubm.sh \
#     --cmd "$train_cmd" $numGaussUBM \
#     ${data_dir}/train ${data_dir}/langp/tri5_ali ${exp_dir}/tri5_ali ${exp_dir}/ubm5
#   touch ${exp_dir}/ubm5/.done
# fi

# if [ ! -f ${exp_dir}/sgmm5/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/sgmm5 on" `date`
#   echo ---------------------------------------------------------------------
#   steps/train_sgmm2.sh \
#     --cmd "$train_cmd" $numLeavesSGMM $numGaussSGMM \
#     ${data_dir}/train ${data_dir}/langp/tri5_ali ${exp_dir}/tri5_ali ${exp_dir}/ubm5/final.ubm ${exp_dir}/sgmm5
#   #steps/train_sgmm2_group.sh \
#   #  --cmd "$train_cmd" "${sgmm_group_extra_opts[@]-}" $numLeavesSGMM $numGaussSGMM \
#   #  data/train data/lang exp/tri5_ali exp/ubm5/final.ubm exp/sgmm5
#   touch ${exp_dir}/sgmm5/.done
# fi

# ################################################################################
# # Ready to start discriminative SGMM training
# ################################################################################

# if [ ! -f ${exp_dir}/sgmm5_ali/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/sgmm5_ali on" `date`
#   echo ---------------------------------------------------------------------
#   steps/align_sgmm2.sh \
#     --nj $train_nj --cmd "$train_cmd" --transform-dir ${exp_dir}/tri5_ali \
#     --use-graphs true --use-gselect true \
#     ${data_dir}/train ${data_dir}/lang ${exp_dir}/sgmm5 ${exp_dir}/sgmm5_ali

#   local/reestimate_langp.sh --cmd "$train_cmd" --unk "$oovSymbol" \
#     ${data_dir}/train ${data_dir}/lang ${data_dir}/local \
#     ${exp_dir}/sgmm5_ali ${data_dir}/local/dictp/sgmm5 ${data_dir}/local/langp/sgmm5 ${data_dir}/langp/sgmm5

#   touch ${exp_dir}/sgmm5_ali/.done
# fi

# if [ ! -f ${exp_dir}/sgmm5_denlats/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/sgmm5_denlats on" `date`
#   echo ---------------------------------------------------------------------
#   steps/make_denlats_sgmm2.sh \
#     --nj $train_nj --sub-split $train_nj "${sgmm_denlats_extra_opts[@]}" \
#     --beam 10.0 --lattice-beam 6 --cmd "$decode_cmd" --transform-dir ${exp_dir}/tri5_ali \
#     ${data_dir}/train ${data_dir}/langp/sgmm5 ${exp_dir}/sgmm5_ali ${exp_dir}/sgmm5_denlats
#   touch ${exp_dir}/sgmm5_denlats/.done
# fi


# if [ ! -f ${exp_dir}/sgmm5_mmi_b0.1/.done ]; then
#   echo ---------------------------------------------------------------------
#   echo "Starting ${exp_dir}/sgmm5_mmi_b0.1 on" `date`
#   echo ---------------------------------------------------------------------
#   steps/train_mmi_sgmm2.sh \
#     --cmd "$train_cmd" "${sgmm_mmi_extra_opts[@]}" \
#     --drop-frames true --transform-dir ${exp_dir}/tri5_ali --boost 0.1 \
#     ${data_dir}/train ${data_dir}/langp/sgmm5 ${exp_dir}/sgmm5_ali ${exp_dir}/sgmm5_denlats \
#     ${exp_dir}/sgmm5_mmi_b0.1
#   touch ${exp_dir}/sgmm5_mmi_b0.1/.done
# fi

echo ---------------------------------------------------------------------
echo "Finished successfully on" `date`
echo ---------------------------------------------------------------------

exit 0
