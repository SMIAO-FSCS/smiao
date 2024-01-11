from adapter_spos.spos.sp1_train_super_adapter_quickly_valid import run_train_supernet
from adapter_spos.spos.sp2_random_search_acc import run_search
from adapter_spos.spos.sp3_retrain_best_path import run_retrain
from spos.sp3_retrain_best_path_acc import run_retrain_acc
from .spos_adaptor_inference import mrr_inference
from .spos_mrr import get_mrr
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def spos_run(lang, num_epochs, batch_size, train_num_list, train_file_path,
             valid_file_path, tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir,
             adaptor_save_dir, infer_file_path, output_infer_file):
    set_seed(1)

    ########## STAGE 1: 训练超网 ###########
    for train_num in train_num_list:  # 训练数据量
        for lr in [6e-4]:
            # train
            run_train_supernet(train_num, lr, train_file_path, valid_file_path, num_epochs, batch_size,
                               tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir)

        print()
        print("训练超网结束")
        print("*" * 60)

        ########## STAGE 2: 搜索最佳路径 ###########
        set_seed(1)
        search_num = 100
        max_acc, best_path = run_search(search_num, tokenizer_path, base_model_path, best_super_adapter_dir,
                                        valid_file_path)
        print("最佳路径: max acc {}, best path {}".format(max_acc, best_path))

        print()
        print("搜索最佳路径结束")
        print("*" * 60)

        ########## STAGE 3: 重新训练搜索到的最佳路径 ###########
        set_seed(1)
        re_num_epochs = 10
        re_batch_size = 64
        # adaptor_save_dir = "../../../save_model/" + lang + "/" + str(lang) + "_adaptor"  # 训练好的最终的adapter的保存位置
        # infer_file_path = "../../../data/test/" + lang + "/batch_0.txt"  # 推理文件目录
        # output_infer_file = "../../../results/" + lang + "/adaptor_batch_0.txt"  # 推理结果目录

        # init_adapter_path = "./random_int_adapter"

        for lr in [3e-3, 1e-3, 8e-4, 6e-4]:
            for re_batch_size in [32]:
                #train
                # open it, use the MRR to determine best model
                # run_retrain(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file, best_path, init_adapter_path)
                # or open it, use the ACC to determine best model
                run_retrain_acc(train_num, lr, lang, adaptor_save_dir, train_file_path, infer_file_path, output_infer_file,
                            best_path, init_adapter_path,
                            re_num_epochs, re_batch_size, tokenizer_path, base_model_path)

                #inference
                mrr_inference(base_model_path, tokenizer_path, adaptor_save_dir, infer_file_path, output_infer_file, 0)

                # get result
                get_mrr(lang)

                print("lr {}, train_num {}, re_batch_size {}".format(lr, train_num, re_batch_size))

if __name__ == '__main__':
    # 配置
    lang = "sql"
    num_epochs = 20
    batch_size = 64
    train_num_list = [14000]

    # 数据
    train_file_path = "../../../data/train_valid/" + lang + "/train.txt"  # 训练文件目录
    valid_file_path = "../../../data/train_valid/" + lang + "/valid.txt"  # valid文件目录
    # 模型
    tokenizer_path = "../../../graph_code_bert"  # 分词器
    base_model_path = "../initial_base_model_adapter/java_model_sqlvalid_0.95hard100000_5e-06"  # 基础模型
    init_adapter_path = "../initial_base_model_adapter/java_adapter_sqlvalid_0.95hard100000_5e-06"  # 初始的adapter

    # 保存
    best_super_adapter_dir = "./{}_best_super_adapter".format(lang)  # 保存最佳性能超网的权重

    adaptor_save_dir = "../../../save_model/" + lang + "/" + str(lang) + "_adaptor"  # 训练好的最终的adapter的保存位置
    infer_file_path = "../../../data/test/" + lang + "/batch_0.txt"  # 推理文件目录
    output_infer_file = "../../../results/" + lang + "/adaptor_batch_0.txt"  # 推理结果目录

    spos_run(lang, num_epochs, batch_size, train_num_list, train_file_path,
             valid_file_path, tokenizer_path, base_model_path, init_adapter_path, best_super_adapter_dir,
             adaptor_save_dir, infer_file_path, output_infer_file)