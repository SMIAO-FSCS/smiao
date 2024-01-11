import torch
import geomloss
import ot

def compute_wasserstein_distance(matrix1, matrix2):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



    matrix1 = matrix1.to(device)
    matrix2 = matrix2.to(device)

    # 创建一个SamplesLoss对象来计算Wasserstein距离
    # loss = geomloss.SamplesLoss(loss='sinkhorn', p=1, cost= geomloss.utils.distances)
    loss = geomloss.SamplesLoss(loss='sinkhorn')

    # 计算Wasserstein距离
    wd = loss(matrix1, matrix2)

    return wd.item()


if __name__ == '__main__':
    # 创建两个矩阵
    matrix1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    matrix2 = torch.tensor([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

    wd = compute_wasserstein_distance(matrix1, matrix2)

    print("Wasserstein Distance:", wd)

