import numpy as np
import torch
import geomloss


def compute_mmd_distance(matrix1, matrix2):
    # 将数据移动到GPU上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # matrix1 = matrix1.to(device)
    # matrix2 = matrix2.to(device)

    # 创建一个SamplesLoss对象来计算MMD距离
    loss = geomloss.SamplesLoss(loss='energy', backend='tensorized', blur=0.05)

    # 计算MMD距离
    mmd_distance = loss(matrix1, matrix2)

    return mmd_distance.item()


def compute_mmd_distance_2(matrix1, matrix2):
    mean_1 = torch.mean(matrix1, axis=0)
    mean_2 = torch.mean(matrix2, axis=0)

    mmd_distance = torch.norm(mean_1-mean_2)

    return mmd_distance


if __name__ == '__main__':
    # 创建两个矩阵
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    mmd_distance = compute_mmd_distance_2(matrix1, matrix2)

    print("MMD Distance:", mmd_distance.item())
