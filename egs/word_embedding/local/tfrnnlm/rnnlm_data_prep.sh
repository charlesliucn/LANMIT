#!/bin/bash

# This script prepares the data directory used for TensorFlow based RNNLM traiing
# it prepares the following files in the output-directory
# 1. $dir/wordlist.rnn.final : wordlist for RNNLM
#    format of this file is like the following:
#        0   The
#        1   a
#        2   is
#        ....
#    note that we don't reserve the 0 id for any special symbol
# 2. $dir/{train/valid} : the text files, with each sentence in a line

# 3. $dir/unk.probs : this file provides information for distributing unk probs
#    among all the unk words, in rnnlm-rescoring.  If provided, the
#    probability for <unk> would be porportionally distributed among all unk words
#
#    It is called unk.probs to be consistent with rnnlm-rescoring scripts with
#    Mikolov's and Yandex's toolkits, but you could simply provide the count instead, as
#    the binary would auto-normalize the counts into probabilities
#    the format of this file is like the following:
#         some-rare-word-1  0.0003
#         some-rare-word-2  0.0004
#         ...

set -e

train_text=data/train/text
dev_text=data/dev/text

if [ $# != 1 ]; then
   echo "Usage: $0 <dest-dir>"
   echo "For details of what the script does, see top of script file"
   exit 1;
fi

dir=$1
srcdir=data/local/dict

mkdir -p $dir

echo "-----------------------------------"
echo "     Train/Dev Data Preparation    "
echo "-----------------------------------"
cat $srcdir/lexicon.txt | awk '{print $1}' | sort -u | grep -v -w '!SIL' > $dir/wordlist.all

# Get training data with OOV words (w.r.t. our current vocab) replaced with <unk>,
# as well as adding </s> symbols at the end of each sentence
cat $train_text | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}' | sed 's=$= </s>=g' \
  | utils/shuffle_list.pl | gzip -c > $dir/train.gz

cat $dev_text | awk -v w=$dir/wordlist.all \
  'BEGIN{while((getline<w)>0) v[$1]=1;}
  {for (i=2;i<=NF;i++) if ($i in v) printf $i" ";else printf "<unk> ";print ""}' | sed 's=$= </s>=g' \
  | utils/shuffle_list.pl | gzip -c > $dir/dev.gz

echo "Getting train and validation sets."

gunzip -c $dir/dev.gz > $dir/valid.in # validation data
gunzip -c $dir/train.gz > $dir/train.in # training data

cat $dir/train.in $dir/wordlist.all | \
  awk '{ for(x=1;x<=NF;x++) count[$x]++; } END{for(w in count){print count[w], w;}}' | \
  sort -nr > $dir/unigram.counts

total_nwords=`wc -l < $dir/unigram.counts`

# the wordlist.rnn file is just a wordlist - i.e. with a word on each lien
# wordlist.rnn.id has [word-id] [word] on each line, with [word-id] being consecutive integers
# this will not be the final wordlist we use because we need to add <unk> symbol
cat $dir/unigram.counts | awk '{print $2}' | tee $dir/wordlist.rnn | awk '{print NR-1, $1}' > $dir/wordlist.rnn.id

for type in train valid; do
  # replacing every word that does not appear in the worlist.rnn file with a <unk> symbol
  cat $dir/$type.in | awk -v w=$dir/wordlist.rnn 'BEGIN{while((getline<w)>0)d[$1]=1}{for(i=1;i<=NF;i++){if(d[$i]==1){s=$i}else{s="<unk>"} printf("%s ",s)} print""}' > $dir/$type
done

echo "-----------------------------------"
echo "     Data Preparation Finished!    "
echo "-----------------------------------"

cp $dir/wordlist.rnn $dir/wordlist.rnn.final

if ! grep -w '<unk>' $dir/wordlist.rnn.final >/dev/null; then
  echo "<unk>" >> $dir/wordlist.rnn.final
fi

echo "-----------------------------------"
echo "  Vocabulary Preparation Finished! "
echo "-----------------------------------"
