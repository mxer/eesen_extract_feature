#!/bin/bash

# Copyright 2015  Yajie Miao    (Carnegie Mellon University)
# Apache 2.0

# This script trains acoustic models based on CTC and using SGD. 

## Begin configuration section
train_tool=train-ctc-parallel  # the command for training; by default, we use the
                # parallel version which processes multiple utterances at the same time 

# configs for multiple sequences
num_sequence=5           # during training, how many utterances to be processed in parallel
valid_num_sequence=10    # number of parallel sequences in validation
frame_num_limit=1000000  # the number of frames to be processed at a time in training; this config acts to
         # to prevent running out of GPU memory if #num_sequence very long sequences are processed;the max
         # number of training examples is decided by if num_sequence or frame_num_limit is reached first. 

# learning rate
learn_rate=0.0001        # learning rate
momentum=0.9             # momentum

# learning rate schedule
max_iters=25             # max number of iterations
min_iters=               # min number of iterations
start_epoch_num=1        # start from which epoch, used for resuming training from a break point

start_halving_inc=0.5    # start halving learning rates when the accuracy improvement falls below this amount
end_halving_inc=0.1      # terminate training when the accuracy improvement falls below this amount
halving_factor=0.5       # learning rate decay factor
halving_after_epoch=1    # halving bcomes enabled after this many epochs

# logging
report_step=100          # during training, the step (number of utterances) of reporting objective and accuracy
verbose=1


# feature configs
sort_by_len=true         # whether to sort the utterances by their lengths
min_len=0                # minimal length of utterances to consider

norm_vars=true # whether to apply variance normalization when we do cmvn
add_deltas=true          # whether to add deltas
copy_feats=true # whether to copy features into a local dir (on the GPU machine)
feats_tmpdir=  # the tmp dir to save the copied features, when copy_feats=true

# status of learning rate schedule; useful when training is resumed from a break point
cvacc=0
halving=0

## End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-tr> <data-cv> <exp-dir>"
   echo " e.g.: $0 data/train_tr data/train_cv exp/train_phn"
   exit 1;
fi

data_tr=$1
data_cv=$2
dir=$3

#音素相关的文件夹
phn_dir="data/lang_phn/"
units_file=$phn_dir"units.txt"

mkdir -p ${dir}/log

for f in ${data_tr}/feats.scp ${data_cv}/feats.scp ${dir}/labels.tr.gz ${dir}/labels.cv.gz; do
  [ ! -f ${f} ] && echo "train_ctc_parallel.sh: no such file $f" && exit 1;
done

## Setup features
if ${sort_by_len}; then
  feat-to-len scp:${data_tr}/feats.scp ark,t:- | awk '{print $2}' > ${dir}/len.tmp || exit 1;
  paste -d " " ${data_tr}/feats.scp ${dir}/len.tmp | sort -k3 -n - | awk -v m=${min_len} '{ if ($3 >= m) {print $1 " " $2} }' > ${dir}/train.scp || exit 1;
  feat-to-len scp:${data_cv}/feats.scp ark,t:- | awk '{print $2}' > ${dir}/len.tmp || exit 1;
  paste -d " " ${data_cv}/feats.scp ${dir}/len.tmp | sort -k3 -n - | awk '{print $1 " " $2}' > ${dir}/cv.scp || exit 1;
  rm -f ${dir}/len.tmp
else
  cat ${data_tr}/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > ${dir}/train.scp
  cat ${data_cv}/feats.scp | utils/shuffle_list.pl --srand ${seed:-777} > ${dir}/cv.scp
fi

apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- | \
add-deltas  --delta-order=2 ark:- ark,t:exp/train_phn_l2_c120/train2.ark

apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- | \
add-deltas  --delta-order=2 ark:- ark,t:exp/train_phn_l2_c120/cv2.ark


#feats_tr="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_tr/utt2spk scp:$data_tr/cmvn.scp scp:$dir/train.scp ark:- |"
#feats_cv="ark,s,cs:apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp scp:$dir/cv.scp ark:- |"

# Save the features to a local dir on the GPU machine. On Linux, this usually
# points to /tmp
#if ${copy_feats}; then
#  tmp_dir=$(mktemp -d ${feats_tmpdir});
#  copy-feats "$feats_tr" ark,scp,t:${tmp_dir}/train.ark,${dir}/train_local.scp || exit 1;
#  copy-feats "$feats_cv" ark,scp,t:${tmp_dir}/cv.ark,${dir}/cv_local.scp || exit 1;
#  trap "echo \"Removing features tmpdir ${tmp_dir} @ $(hostname)\"; ls ${tmp_dir}; rm -r ${tmp_dir}" EXIT
#fi

#if ${add_deltas}; then
#  rm -rf exp/train_phn_l2_c120/train2.ark
#  rm -rf exp/train_phn_l2_c120/cv2.ark
#  add-deltas ark:${tmp_dir}/train.ark ark,t:exp/train_phn_l2_c120/train2.ark || exit 1;
#  add-deltas ark:${tmp_dir}/cv.ark ark,t:exp/train_phn_l2_c120/cv2.ark || exit 1;
#fi

#echo "tempdir is "${tmp_dir}
## End of feature setup

gunzip -c ${dir}/labels.tr.gz > ${dir}/labels.tr
gunzip -c ${dir}/labels.cv.gz > ${dir}/labels.cv

cat ${dir}/labels.tr   | cut -d' ' -f 2- | awk '{for( i=1;i<=NF;i++){count[$i]++;}}END{print(length(count))}' > ${dir}/tr.lable.count
cat ${dir}/labels.cv   | cut -d' ' -f 2- | awk '{for( i=1;i<=NF;i++){count[$i]++;}}END{print(length(count))}' > ${dir}/cv.lable.count

cat ${dir}/labels.tr   |  awk '{for(i=2;i<=NF;i++){if(a<$i){a=$i;}}}END{print a;}' >${dir}/tr.lable.count.max
cat ${dir}/labels.cv   |  awk '{for(i=2;i<=NF;i++){if(a<$i){a=$i;}}}END{print a;}' >${dir}/cv.lable.count.max


#因为label会出现，有的label在训练数据中不存在，
#在给ctc的时候， 会出现lable的空隙， 可能会对结果有一点影响，
#这里重新生成一个label的映射
cat ${dir}/labels.tr   | cut -d' ' -f 2- | awk '{for( i=1;i<=NF;i++){label[$i]=1;}}END{for(i in label){print i;}}' | sort -k1n > ${dir}/tr.lable.list
cat ${dir}/labels.cv   | cut -d' ' -f 2- | awk '{for( i=1;i<=NF;i++){label[$i]=1;}}END{for(i in label){print i;}}' | sort -k1n > ${dir}/cv.lable.list

tr_label_max=`cat ${dir}/tr.lable.count.max`
cv_label_max=`cat ${dir}/cv.lable.count.max`

#如果cv的最大的值大于tr的最大的值的话,可能在验证数据的时候，出现出现超出index的错误,
#这里需要手动处理下，但是出现的几率很小
if [ $tr_label_max -ne $cv_label_max ]; then
  echo "Error: train label max not equal to cv label max"
fi

#生成map的映射， 第一列是原来的值， 第二列是新的值
cat ${dir}/tr.lable.list | awk '{print $1 " " NR-1}' > ${dir}/tr.lable.map

wc -l ${dir}/tr.lable.map | awk '{print $1;}' > ${dir}/tr.lable.map.count

awk '{
  if(ARGIND==1){
    units[$2] = $1
  }
  if(ARGIND==2){
    print units[$1],$2;
  }
}' $units_file ${dir}/tr.lable.map  > ${dir}/tr.units.use.txt

