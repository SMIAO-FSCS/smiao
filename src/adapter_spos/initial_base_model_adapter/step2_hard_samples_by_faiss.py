import os
import random

import faiss  # 使Faiss可调用
import pickle

import numpy as np
import pandas as pd
import time

import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def read_data(file_path):
    assert os.path.isfile(file_path)
    print("read data file at:", file_path)

    with open(file_path, encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

    text_lines = []
    code_lines = []
    labels = []

    for line in lines:
        temp_line = line.split("<CODESPLIT>")
        if (len(temp_line)) == 5:  # 确保<CODESPLIT>分开的每个部分都有值，不是Null
            if (str(temp_line[0]) == "1"):  # 只要正例
                text_lines.append(temp_line[-2].lower())  # 注释
                code_lines.append(temp_line[-1].lower())  # 代码
                labels.append(int(temp_line[0]))

    print("注释和代码总行数:", len(text_lines), len(code_lines))

    return text_lines, code_lines

def write_data_to_file(output_file, new_all_labels, new_all_quries, new_all_codes):
    with open(output_file, "w") as writer:
        for label, query, code in zip(new_all_labels, new_all_quries, new_all_codes):
            writer.write(str(label) + "<CODESPLIT>URL<CODESPLIT>func_name" + '<CODESPLIT>' + '<CODESPLIT>'.join([query, code]) + '\n')


def text_nearst(text_pkl, topK):
    faiss.seed_random = 1
    # faiss.omp_set_num_threads(1)

    df_text = pd.DataFrame(text_pkl).astype('float32')

    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray

    print('开始降低维度')
    time_start = time.time()
    mat = faiss.PCAMatrix(768, 256)
    mat.rand_seed = 1
    mat.train(df_text)
    assert mat.is_trained
    df_text = mat.apply_py(df_text)
    time_end = time.time()
    print('PCA 耗时', time_end - time_start, 's')
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法2
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatL2(256)
    index = faiss.index_cpu_to_gpu(res, 0, index_flat)
    index.add(df_text)
    print(index.is_trained)

    # 检索
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("text near examples")
    print(I[:10])
    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text = []
    for line in I:
        # print(line[-1])
        nearest_text.append(line[-1])  # line[0]是其本身，取排在第二个相似的

    return nearest_text


def code_nearst(text_pkl, topK):
    df_text = pd.DataFrame(text_pkl).astype('float32')
    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray
    # python数据分为行连续和列连续或者都不连续，指的是数据在内存的存储是否连续，np.ascontiguousarray()可以把不连续的变成连续的，之后就可以写入了。

    # 建立索引   方法1
    # https://zhuanlan.zhihu.com/p/357414033
    # dim, measure = 768, faiss.METRIC_L2  # 33.525 s
    # param = 'Flat'
    # index = faiss.index_factory(dim, param, measure)
    # index.add(df_text)  # 将向量库中的向量加入到index中

    # 建立索引   方法2
    dim, measure = 768, faiss.METRIC_L2 #3s
    param = 'HNSW64'
    index = faiss.index_factory(dim, param, measure)
    index.seed = 1
    print(index.is_trained)  # 此时输出为True
    index.add(df_text) #3.10 s

    # 检索
    time_start = time.time()
    # xq = np.array([df_text[-1].astype('float32')])
    xq = np.array(df_text)
    k = topK  # topK的K值
    D, I = index.search(xq, k)  # xq为待检索向量，返回的I为每个待检索query最相似TopK的索引list，D为其对应的距离
    print("code near examples")
    print(I[:15])
    # print(D[-5:])
    time_end = time.time()
    print('time cost', time_end - time_start, 's')

    # 构建困难负样本试试？或者构建又简单到难的试试，可以调整这两者的比例
    nearest_text = []
    for line in I:
        # print(line[-1])
        nearest_text.append(line[-1])  # line[0]是其本身，取排在第二个相似的

    return nearest_text


#为什么不去构造困难反例呢？检索慢，而且可能检索不到，构造的反例，贴近测试集效果会更好
def arrange_hard_samples(lang, topK, in_file_path, output_file, simple_proportion=1):
    set_seed(1)

    # lang  = "java"

    code_based = 0
    query_based = 1
    # topK = 5  #取排在第几位的近似值

    text_pkl_file = open("text.pkl", 'rb') #query和code的cls位置提取的向量
    # code_pkl_file = open("code.pkl", 'rb')

    # in_file_path = "../../../data/train_valid/" + lang + "/train.txt"   #输入的数据，这个会被用于构造正样本
    # output_file = "../../../data/train_valid/" + lang + "/hardonce_topk"+ str(topK) + "unixcoder.txt"  # 输出文件的位置

    text_pkl = pickle.load(text_pkl_file)
    # code_pkl = pickle.load(code_pkl_file)

    nearest_text = text_nearst(text_pkl["mean"], topK)
    # nearset_code = code_nearst(code_pkl["mean"], topK)

    #拿到了最相似的query，替换该query至最近的
    # print(nearest)

    #拿出query和code，进行pair数据
    #然后写入文件

    text_lines, code_lines = read_data(in_file_path)

    new_all_labels = []
    new_all_quries = []
    new_all_codes = []

    wrong_shard_query_num = 0
    hard_num = 0
    simple_num = 0

    random.seed(1)
    for text, code, near_text_index in zip(text_lines, code_lines, nearest_text):
        if (query_based):
            if (str(text).lower().strip() != str(text_lines[near_text_index]).lower().strip()):
                rand_num = random.random()
                if rand_num < simple_proportion: #困难样本的比例，默认全困难
                    #构造的反例
                    new_all_labels.append(0)
                    new_all_quries.append(text_lines[near_text_index])
                    new_all_codes.append(code)

                    hard_num += 1

                else:
                    new_all_labels.append(0)
                    new_all_quries.append(random.choice(text_lines))
                    new_all_codes.append(code)

                    simple_num += 1

                # 构造的正例
                new_all_labels.append(1)
                new_all_quries.append(text)
                new_all_codes.append(code)

            else:
                wrong_shard_query_num += 1

    write_data_to_file(output_file, new_all_labels, new_all_quries, new_all_codes)


    print("out_file_path {}, wrong_shard_query_num {}".format(output_file, wrong_shard_query_num))
    print("hard_num {}, simple_num {}".format(hard_num, simple_num))


if __name__ == '__main__':
    set_seed(1)

    lang  = "solidity"
    topK = 5  # 取排在第几位的近似值
    ptm_name = "graph_code_bert"

    simple_proportion = 0.3

    in_file_path = "../../../data/train_valid/" + lang + "/train_confident.txt"   #输入的数据，这个会被用于构造正样本
    output_file = "../../../data/train_valid/" + lang + "/" + str(simple_proportion) + "_hardonce_topk" + str(topK) + str(ptm_name) + ".txt"  # 输出文件的位置

    arrange_hard_samples(lang, topK, in_file_path, output_file, simple_proportion=simple_proportion)






