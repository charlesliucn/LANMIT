#!/bin/bash

tag_percentage=0.1
data_dir=data_bnf
web_datafile=

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

lexicon=${data_dir}/local_web/lexicon.txt

./local/check_tools.sh || exit 1

[ -d ${data_dir}/train_web ] && rm -rf ${data_dir}/train_web
[ -d ${data_dir}/srilm_web ] && rm -rf ${data_dir}/srilm_web
# [ -d ${data_dir}/lang_web  ] && rm -rf ${data_dir}/lang_web

#Preparing dev10h and train directories
if [ ! -f ${data_dir}/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------

    local/make_corpus_subset.sh "$train_data_dir" "$train_data_list" ./${data_dir}/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./${data_dir}/raw_train_data`
    touch ${data_dir}/raw_train_data/.done
else 
    echo ---------------------------------------------------------------------
    echo "Already Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------
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
else
  echo ---------------------------------------------------------------------
  echo "Already Subsetting the DEV10H set"
  echo ---------------------------------------------------------------------	
fi
dev10h_data_dir=`utils/make_absolute.sh ./${data_dir}/raw_dev10h_data`

mkdir -p ${data_dir}/lang_web
if [[ ! -f ${data_dir}/lang_web/L.fst || ${data_dir}/lang_web/L.fst -ot $lexicon ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating L.fst etc in ${data_dir}/lang_web on" `date`
  echo ---------------------------------------------------------------------
  utils/prepare_lang.sh --phone-symbol-table ${data_dir}/lang/phones.txt \
    --share-silence-phones true \
    ${data_dir}/local_web $oovSymbol ${data_dir}/local_web/tmp.lang_web ${data_dir}/lang_web
fi

if [[ ! -f ${data_dir}/srilm_web/lm.gz || ${data_dir}/srilm_web/lm.gz -ot ${data_dir}/train_web/text ]]; then
  echo ---------------------------------------------------------------------
  echo "Training SRILM web language models on" `date`
  echo ---------------------------------------------------------------------
  local/train_lms_srilm_addweb.sh  --oov-symbol "$oovSymbol"\
    --words-file ${data_dir}/lang_web/words.txt \
    --train-text ${data_dir}/${web_datafile} \
    --dev-text ${data_dir}/dev10h_bnf/text ${data_dir} ${data_dir}/srilm_web
fi

if [[ ! -f ${data_dir}/lang_web/G.fst || ${data_dir}/lang_web/G.fst -ot ${data_dir}/srilm_web/lm.gz ]]; then
  echo ---------------------------------------------------------------------
  echo "Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
  local/arpa2G.sh ${data_dir}/srilm_web/lm.gz ${data_dir}/lang_web ${data_dir}/lang_web
  echo ---------------------------------------------------------------------
  echo "Done Creating G.fst on " `date`
  echo ---------------------------------------------------------------------
fi

