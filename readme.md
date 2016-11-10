# 需要设置的地方
1. local/tedlium_prepare_data.sh
   1.修改data_dir,指向格式整理好的目录
      data_dir=/asrDataCenter/dataCenter/aligned_audios/batch1_hours_full
       目标目录下的结构:
        -train
            -stm
              -所有的stm文件
            -wav
              -所有的wav文件
        -test
            -stm
              -所有的测试的stm文件，随便放几个就可以
            -wav
              -所有的测试的wav文件， 和test下的stm文件要对应起来

    2. 修改dicFile的值，指向词典的文件名:
      dicFile=/asrDataCenter/dataCenter/aligned_audios/batch1_hours_full/lm/dic4LM.dic
       
2. local/tedlium_prepare_phn_dict.sh, 修改语言模型对应的词典
   1. srcdict=/home/zjl/dataCenter/asr/tedlium/cantab-TEDLIUM/cantab-TEDLIUM.dct
   
3. local/tedlium_decode_graph.sh 修改语言模型的目录
   1. arpa_lm=/home/zjl/dataCenter/asr/tedlium/cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3.gz



然后就执行运行根目录下的data_prepare_mfcc.sh生成mfcc特征， 运行data_prepare_fbank.sh生成fbank的特征。

