# 需要设置的地方
1. prepare_data.sh
   1.修改data_dir,指向格式整理好的目录
       目标目录结构
       data_dir
        train
            stm
            wav
        test
            stm
            wav
    2. 修改词典的文件名:
       join_suffix.py ${data_dir}/TEDLIUM.*.dic(LINE 50)
       
2. prepare_phn_dict.sh, 修改语言模型对应的词典
   1. srcdict=/home/zhangjl/dataCenter/asr/tedlium/cantab-TEDLIUM/cantab-TEDLIUM.dct
   
3. decode_graph.sh 修改语言模型的目录
   1. arpa_lm=/home/zhangjl/dataCenter/asr/tedlium/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz

4. 修改网络结构:run_ctc_phn.sh
   1. FBank Feature Generation 
      ```
      lstm_layer_num=5     # number of LSTM layers                                                                                    
      lstm_cell_dim=320    # number of memory cells in every LSTM layer 
      ```
