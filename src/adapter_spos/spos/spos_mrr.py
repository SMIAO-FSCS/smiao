# -*- coding: utf-8 -*-
# Copyright (c) Microsoft Corporation. 
# Licensed under the MIT license.

import os
import numpy as np
from more_itertools import chunked
import argparse

PROJECT_ROOT = os.getcwd()[:-44]

def get_mrr(lang):
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_batch_size', type=int, default=1000)
    # args = parser.parse_args()
    languages = [lang]
    MRR_dict = {}
    all_result = []
    for language in languages:
        file_dir = '../../results/{}'.format(language)
        # file_dir = "results/temp/" #只是到文件夹级别，没有到文件夹中的文件
        ranks = []
        max_array = []
        num_batch = 0
        for file in sorted(os.listdir(file_dir)):
            print(os.path.join(file_dir, file)) #扫描该文件夹下的文件，并进行计算
            with open(os.path.join(file_dir, file), encoding='utf-8') as f:
                batched_data = chunked(f.readlines(), 1000)
                for batch_idx, batch_data in enumerate(batched_data):
                    num_batch += 1
                    correct_score = float(batch_data[batch_idx].strip().split('<CODESPLIT>')[-1])
                    scores = np.array([float(data.strip().split('<CODESPLIT>')[-1]) for data in batch_data])

                    max_value = np.argmax(scores)

                    rank = np.sum(scores >= correct_score)
                    all_result.append(rank)
                    ranks.append(rank)
                    max_array.append(max_value)

        mean_mrr = np.mean(1.0 / np.array(ranks))

        MRR_dict[language] = mean_mrr
    for key, val in MRR_dict.items():
        print("{} mrr: {}".format(key, val))

    rk_1 = np.sum(np.array(all_result) == 1) / len(all_result)
    rk_5 = np.sum(np.array(all_result) <= 5) / len(all_result)
    rk_10 = np.sum(np.array(all_result) <= 10) / len(all_result)

    print("all_rk_1: ", rk_1)
    print("all_rk_5: ", rk_5)
    print("all_rk_10: ", rk_10)

    return mean_mrr

if __name__ == '__main__':
    get_mrr('sql')

