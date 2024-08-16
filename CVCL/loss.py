import time  # 导入time模块，用于处理时间相关操作
import torch  # 导入PyTorch库
import torch.nn as nn  # 从PyTorch中导入神经网络模块
import numpy as np  # 导入NumPy库，用于科学计算
import math  # 导入math模块，用于数学运算
from metrics import *  # 从metrics模块中导入所有内容
import torch.nn.functional as F  # 从PyTorch中导入功能模块
from torch.nn.functional import normalize  # 从PyTorch中导入normalize函数
from dataprocessing import *  # 从dataprocessing模块中导入所有内容


# 定义一个深度多视图聚类损失类，继承自torch.nn.Module
class DeepMVCLoss(nn.Module):
    # 初始化函数，接受样本数量和聚类数量作为参数
    def __init__(self, num_samples, num_clusters):
        super(DeepMVCLoss, self).__init__()  # 调用父类的初始化函数
        self.num_samples = num_samples  # 样本数量
        self.num_clusters = num_clusters  # 聚类数量

        self.similarity = nn.CosineSimilarity(dim=2)  # 余弦相似度计算，沿第2维度计算
        self.criterion = nn.CrossEntropyLoss(reduction="sum")  # 交叉熵损失函数，使用总和作为损失的减少方式

    # 创建一个掩码矩阵，用于掩盖相关样本的对角线元素
    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))  # 创建一个N x N的全1矩阵
        mask = mask.fill_diagonal_(0)  # 将对角线元素置为0
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0  # 掩盖相关样本对
            mask[N // 2 + i, i] = 0  # 掩盖相关样本对
        mask = mask.bool()  # 将掩码转换为布尔类型

        return mask  # 返回掩码矩阵

    # 计算概率分布的前向传播
    def forward_prob(self, q_i, q_j):
        q_i = self.target_distribution(q_i)  # 计算目标分布
        q_j = self.target_distribution(q_j)  # 计算目标分布

        p_i = q_i.sum(0).view(-1)  # 计算每个聚类的概率并展平
        p_i /= p_i.sum()  # 归一化概率
        ne_i = (p_i * torch.log(p_i)).sum()  # 计算熵

        p_j = q_j.sum(0).view(-1)  # 计算每个聚类的概率并展平
        p_j /= p_j.sum()  # 归一化概率
        ne_j = (p_j * torch.log(p_j)).sum()  # 计算熵

        entropy = ne_i + ne_j  # 总熵

        return entropy  # 返回熵

    # 计算标签分布的前向传播
    def forward_label(self, q_i, q_j, temperature_l, normalized=False):
        q_i = self.target_distribution(q_i)  # 计算目标分布
        q_j = self.target_distribution(q_j)  # 计算目标分布

        q_i = q_i.t()  # 转置矩阵
        q_j = q_j.t()  # 转置矩阵
        N = 2 * self.num_clusters  # 样本数量（2倍聚类数量）
        q = torch.cat((q_i, q_j), dim=0)  # 拼接两个分布

        if normalized:
            # 如果归一化，则计算余弦相似度
            sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
        else:
            # 否则，计算内积相似度
            sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)

        sim_i_j = torch.diag(sim, self.num_clusters)  # 计算正样本对的相似度
        sim_j_i = torch.diag(sim, -self.num_clusters)  # 计算正样本对的相似度

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)  # 拼接正样本对相似度
        mask = self.mask_correlated_samples(N)  # 获取掩码矩阵
        negative_clusters = sim[mask].reshape(N, -1)  # 获取负样本对相似度

        labels = torch.zeros(N).to(positive_clusters.device).long()  # 创建标签，标签为0表示正样本对
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)  # 拼接正负样本对相似度
        loss = self.criterion(logits, labels)  # 计算交叉熵损失
        loss /= N  # 平均损失

        return loss  # 返回损失

    # 计算目标分布
    def target_distribution(self, q):
        weight = (q ** 2.0) / torch.sum(q, 0)  # 计算权重，平方分布除以总和
        return (weight.t() / torch.sum(weight, 1)).t()  # 返回目标分布，按列归一化



# import time
# import torch
# import torch.nn as nn
# import numpy as np
# import math
# from metrics import *
# import torch.nn.functional as F
# from torch.nn.functional import normalize
# from dataprocessing import *
#
#
# class DeepMVCLoss(nn.Module):
#     def __init__(self, num_samples, num_clusters):
#         super(DeepMVCLoss, self).__init__()
#         self.num_samples = num_samples
#         self.num_clusters = num_clusters
#
#         self.similarity = nn.CosineSimilarity(dim=2)
#         self.criterion = nn.CrossEntropyLoss(reduction="sum")
#
#     def mask_correlated_samples(self, N):
#         mask = torch.ones((N, N))
#         mask = mask.fill_diagonal_(0)
#         for i in range(N // 2):
#             mask[i, N // 2 + i] = 0
#             mask[N // 2 + i, i] = 0
#         mask = mask.bool()
#
#         return mask
#
#     def forward_prob(self, q_i, q_j):
#         q_i = self.target_distribution(q_i)
#         q_j = self.target_distribution(q_j)
#
#         p_i = q_i.sum(0).view(-1)
#         p_i /= p_i.sum()
#         ne_i = (p_i * torch.log(p_i)).sum()
#
#         p_j = q_j.sum(0).view(-1)
#         p_j /= p_j.sum()
#         ne_j = (p_j * torch.log(p_j)).sum()
#
#         entropy = ne_i + ne_j
#
#         return entropy
#
#     def forward_label(self, q_i, q_j, temperature_l, normalized=False):
#
#         q_i = self.target_distribution(q_i)
#         q_j = self.target_distribution(q_j)
#
#         q_i = q_i.t()
#         q_j = q_j.t()
#         N = 2 * self.num_clusters
#         q = torch.cat((q_i, q_j), dim=0)
#
#         if normalized:
#             sim = (self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / temperature_l).to(q.device)
#         else:
#             sim = (torch.matmul(q, q.T) / temperature_l).to(q.device)
#
#         sim_i_j = torch.diag(sim, self.num_clusters)
#         sim_j_i = torch.diag(sim, -self.num_clusters)
#
#         positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
#         mask = self.mask_correlated_samples(N)
#         negative_clusters = sim[mask].reshape(N, -1)
#
#         labels = torch.zeros(N).to(positive_clusters.device).long()
#         logits = torch.cat((positive_clusters, negative_clusters), dim=1)
#         loss = self.criterion(logits, labels)
#         loss /= N
#
#         return loss
#
#
#     def target_distribution(self, q):
#         weight = (q ** 2.0) / torch.sum(q, 0)
#         return (weight.t() / torch.sum(weight, 1)).t()
