import pickle

import numpy as np
import pandas as pd
import torch

from adapter_spos.measure_data_distance.MMD import compute_mmd_distance, compute_mmd_distance_2
from adapter_spos.measure_data_distance.Wasserstein_Distance import compute_wasserstein_distance


def read_pkl(pkl_file_path):
    text_pkl_file = open(pkl_file_path, 'rb')
    text_pkl = pickle.load(text_pkl_file)

    df_text = pd.DataFrame(text_pkl["mean"]).astype('float32')
    df_text = np.ascontiguousarray(np.array(df_text))  # 转换为nparray

    df_text = torch.tensor(df_text)

    print("{}.shape: {}".format(pkl_file_path, df_text.shape))

    return df_text




if __name__ == '__main__':
    code1_pkl_file_path = "./pkl/solidity_code.pkl"

    for lang in ["ruby","python","php","go","java", "javascript"]:
        code2_pkl_file_path = "./pkl/" + lang + "_code.pkl"

        code_matrix = read_pkl(code1_pkl_file_path)
        query_matrix = read_pkl(code2_pkl_file_path)

        wasserstein_distance = compute_wasserstein_distance(code_matrix, query_matrix)
        print("wasserstein_distance: ", wasserstein_distance)

        # mmd_distance = compute_mmd_distance(code_matrix, query_matrix)
        # mmd_distance_2 = compute_mmd_distance_2(code_matrix, query_matrix)
        # #
        # print("lang: {}, mmd_distance:{}".format(lang, mmd_distance))
        # print("lang: {}, mmd_distance:{}".format(lang, mmd_distance_2))

        print()


