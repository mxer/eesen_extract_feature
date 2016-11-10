#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Kuang.Ru on 2016/8/29
# 该文件是用于将stm数据转化为二进制的特征文本, 作为tensorflow训练模型的输入.
import sys
import os
import logging
import subprocess

try:
  from src.decode import prepare_decode_data as pdd
except ImportError:
  sys.path.append("/home/rukuang/remote_map/asr")
  from src.decode import prepare_decode_data as pdd


def __prepare_environment():
  """设置执行shell的环境变量, 为了以后调用一些Essen的脚本.

  """
  fst_bin_path = "/home/rukuang/software/eesen/src/fstbin/"
  decode_bin_path = "/home/rukuang/software/eesen/src/decoderbin/"
  pwd = sys.path[0]
  utils = os.path.join(pwd, "utils/")
  paths = [pwd, utils, fst_bin_path, decode_bin_path, os.environ["PATH"]]
  os.environ["PATH"] = ":".join(paths)
  os.environ["LC_ALL"] = "C"
  logging.debug("系统PATH变量: " + os.environ["PATH"])


def __prepare_data_and_construct_fst():
  """准备训练数据, 并构建解码WFST.

  """
  subprocess.run("local/tedlium_prepare_data.sh")
  decode_dir = "/asrDataCenter/dataCenter/lm/td/reArrange"
  dict_path = os.path.join(decode_dir, "clearedDct4LMGened.dct")
  pdd.prepare_phn_dict(dict_path, "dict_src")
  pdd.ctc_compile_dict_token("dict_src", "dict_tmp", "dict_target")
  lm_path = os.path.join(decode_dir, "clearedTxt4LMGened.lm3.gz")
  pdd.compose_all_fst("dict_target", lm_path)


def __generate_f_bank_feature():
  """生成fbank特征, 以及cmvn等统计量.

  """
  train_cmd = "run.pl"
  f_bank_dir = "fbank"

  # 生成fbank特征; 每一帧的fbanks默认40维.
  for set in ["train", "test"]:
    make_f_bank_script = [
      "steps/make_fbank.sh", "--cmd", train_cmd, "--nj", "20",
      "data/{0}".format(set), "exp/make_fbank/{0}".format(set), f_bank_dir
    ]

    subprocess.run(make_f_bank_script)
    subprocess.run(["utils/fix_data_dir.sh", "data/{0}".format(set)])

    compute_cmvn_script = ["steps/compute_cmvn_stats.sh", "data/" + set,
                           "exp/make_fbank/" + set, f_bank_dir]

    subprocess.run(compute_cmvn_script)

  # 把整个训练集分割成training (95%)和cross-validation (5%).
  split_script = ["utils/subset_data_dir_tr_cv.sh", "--cv-spk-percent", "5",
                  "data/train", "data/train_tr95", "data/train_cv05"]

  subprocess.run(split_script)


def __get_features_data(train_data_path, dev_data_path, result_dir):
  sort_by_len = True
  min_len = 0
  norm_vars = True
  add_deltas = True
  copy_feats = True
  os.makedirs(os.path.join(result_dir, "log"), exist_ok=True)
  # TODO(Kuang.Ru): 继续改写.


if __name__ == '__main__':
  logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                      level=logging.INFO)

  __prepare_environment()

  # 表明程序执行的阶段. 分成3块, 分别是数据准备, feature生成和训练数据生成
  start_stage = 2

  if start_stage == 0:
    logging.info("数据准备和FST构建.")
    __prepare_data_and_construct_fst()

  if start_stage <= 1:
    logging.info("FBank特征生成.")
    __generate_f_bank_feature()

  if start_stage <= 2:
    logging.info("生成训练数据.")
    lstm_layer_num = 2  # LSTM层数.
    lstm_cell_dim = 120  # 每层的宽度.
    exp_dir = "exp/train_phn_l{0}_c{1}".format(lstm_layer_num, lstm_cell_dim)
    os.makedirs(exp_dir, exist_ok=True)

    # 标签序列; 将词转换为标签索引.
    convert_train_script = (
      """utils/prep_ctc_trans.py dict_target/lexicon_numbers.txt"""
      """data/train_tr95/text "<UNK>" | gzip -c - > {0}/labels.tr.gz"""
    ).format(exp_dir)

    subprocess.run(convert_train_script, shell=True)

    convert_dev_script = (
      """utils/prep_ctc_trans.py dict_target/lexicon_numbers.txt"""
      """data/train_cv05/text "<UNK>" | gzip -c - > {0}/labels.cv.gz"""
    ).format(exp_dir)

    subprocess.run(convert_dev_script, shell=True)

    script = ["steps/train_ctc_parallel.sh", "--add-deltas", "true",
              "--feats-tmpdir", "{0}/XXXXX".format(exp_dir), "data/train_tr95",
              "data/train_cv05", exp_dir]

    subprocess.run(script)
    logging.info("数据生成完成.")
