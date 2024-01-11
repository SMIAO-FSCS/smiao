import random

import numpy as np
import torch

from adapter_spos.initial_base_model_adapter.step1_extract_features_without_adaptor import extract_features_without_adapter
from adapter_spos.initial_base_model_adapter.step2_hard_samples_by_faiss import arrange_hard_samples
from adapter_spos.initial_base_model_adapter.step3_init import init_model_adapter


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_init(train_lang, valid_lang, pre_train_model, ptm_name, num_epochs, batch_size, lr, topK, simple_proportion,
             init_train_num, in_file_path, output_file, valid_file_path):
    set_seed(1)
    ###
    # train_lang = 'java'
    # valid_lang = 'solidity'
    # pre_train_model = "../../../graph_code_bert"
    #
    # ptm_name = "graph_code_bert"
    #
    # num_epochs = 5
    # batch_size = 64
    # lr = 2e-5
    #
    # simple_proportion = 0.85 #困难样本的比例
    # init_train_num = 236176 # init model trainning data num
    ###

    # step 1
    set_seed(1)
    # in_file_path = "../../../data/train_valid/" + train_lang + "/train.txt"
    out_query_features_path = "text.pkl"
    out_code_features_path = "" # not use
    tokenizer_name = pre_train_model

    extract_features_without_adapter(in_file_path, pre_train_model, tokenizer_name, out_query_features_path,
                                     out_code_features_path)

    # step 2
    set_seed(1)
    # topK = 5
    # output_file = "../../../data/train_valid/" + train_lang + "/hardonce_topk" + str(topK) + str(ptm_name) + ".txt"  # 输出文件的位置

    arrange_hard_samples(train_lang, topK, in_file_path, output_file, simple_proportion)

    # step 3
    set_seed(1)

    init_adaptor_save_dir = "./initial_base_model_adapter/" + train_lang + "_adapter_" + valid_lang + "valid_" + str(topK) + "hard" + str(init_train_num) + "_" + str(ptm_name)
    init_basemodel_save_dir = "./initial_base_model_adapter/" + train_lang + "_model_" + valid_lang + "valid_" + str(topK) + "hard" + str(init_train_num) + "_" + str(ptm_name)

    train_file_path = output_file

    # valid_file_path = "../../../data/train_valid/" + valid_lang + "/valid.txt"  # arg1

    init_model_adapter(num_epochs, batch_size, lr, init_adaptor_save_dir, init_basemodel_save_dir, pre_train_model,
                       train_file_path, valid_file_path, init_train_num)

    print(init_adaptor_save_dir)
    print(init_basemodel_save_dir)

    return init_adaptor_save_dir, init_basemodel_save_dir

if __name__ == '__main__':
    set_seed(1)
    ###
    train_lang = 'java'
    valid_lang = 'sql'
    pre_train_model = "../../../graph_code_bert"

    ptm_name = "graph_code_bert"

    num_epochs = 5
    batch_size = 64
    lr = 2e-5

    topK = 5
    simple_proportion = 0  # 困难样本的比例
    init_train_num = 200000  # init model trainning data num
    ###
    in_file_path = "../../../data/train_valid/" + train_lang + "/train.txt"
    output_file = "../../../data/train_valid/" + train_lang + "/hardonce_topk" + str(topK) + str(
        ptm_name) + ".txt"  # 输出文件的位置
    valid_file_path = "../../../data/train_valid/" + valid_lang + "/valid.txt"  # arg1

    run_init(train_lang, valid_lang, pre_train_model, ptm_name, num_epochs, batch_size, lr, topK, simple_proportion,
             init_train_num, in_file_path, output_file, valid_file_path)