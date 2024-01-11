import random

import numpy as np
import torch

from adapter_spos.initial_base_model_adapter.init_main import run_init
from adapter_spos.spos.sp0_main import spos_run


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_seed(1)

    need_retrain_base = True # whether need retrain init process

    ############################## init model
    train_lang = 'java' # init model data
    valid_lang = 'sql' # target data
    pre_train_model = "../../graph_code_bert" # pretrained model

    num_epochs = 5 # init epochs
    batch_size = 64 # init
    lr = 2e-5 # init

    topK = 5 # hard samples rank
    init_train_num = 100000  # init model trainning data num

    ### not need config ###
    hard_proportion = 1
    ptm_name = "graph_code_bert"
    in_file_path = "../../data/train_valid/" + train_lang + "/train.txt"
    output_file = "../../data/train_valid/" + train_lang + "/hardonce_topk" + str(topK) + str(
        ptm_name) + ".txt"  # 输出文件的位置
    valid_file_path = "../../data/train_valid/" + valid_lang + "/valid.txt"  # arg1

    if(need_retrain_base):
        init_adaptor_save_dir, init_basemodel_save_dir = run_init(train_lang, valid_lang, pre_train_model, ptm_name, num_epochs, batch_size, lr, topK, hard_proportion,
                 init_train_num, in_file_path, output_file, valid_file_path)
    else:
        # when need_retrain_base=False, config these
        init_adaptor_save_dir = "./initial_base_model_adapter/init_adapter"
        init_basemodel_save_dir = "./initial_base_model_adapter/init_basemodel"



    ############################ spos train
    set_seed(1)
    # 配置
    lang = valid_lang
    spos_num_epochs = 20 # train supernet epochs
    spos_batch_size = 64 # train supernet batch_size

    train_num_list = [100] # retrain

    ### not need config ###
    train_file_path = "../../data/train_valid/" + lang + "/train.txt"  # 训练文件目录
    valid_file_path = "../../data/train_valid/" + lang + "/valid.txt"  # valid文件目录

    tokenizer_path = pre_train_model # 分词器
    base_model_path = init_basemodel_save_dir  # 基础模型
    init_adapter_path = init_adaptor_save_dir  # 初始的adapter

    best_super_adapter_dir = "./initial_base_model_adapter/{}_best_super_adapter".format(lang)  # 保存最佳性能超网的权重

    adaptor_save_dir = "../../save_model/" + lang + "/" + str(lang) + "_adaptor"  # 训练好的最终的adapter的保存位置
    infer_file_path = "../../data/test/" + lang + "/batch_0.txt"  # 推理文件目录
    output_infer_file = "../../results/" + lang + "/adaptor_batch_0.txt"  # 推理结果目录

    spos_run(lang, spos_num_epochs, spos_batch_size, train_num_list, train_file_path,
             valid_file_path, tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir,
             adaptor_save_dir, infer_file_path, output_infer_file)



