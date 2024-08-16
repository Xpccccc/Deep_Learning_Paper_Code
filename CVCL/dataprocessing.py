import os  # 导入os模块，用于处理文件和目录路径
import random  # 导入random模块，用于生成随机数
import sys  # 导入sys模块，用于与Python解释器交互

import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库，用于科学计算
import scipy.io as sio  # 导入scipy.io模块，用于处理MATLAB文件
from sklearn.preprocessing import MinMaxScaler  # 从sklearn.preprocessing中导入MinMaxScaler，用于数据归一化

from torch.utils.data import Dataset  # 从PyTorch中导入Dataset类，用于创建数据集
# from torch.nn.functional import normalize  # 从PyTorch中导入normalize函数（注释掉了）
from utils import *  # 从utils模块中导入所有内容


# 定义一个多视图数据类，继承自torch.utils.data.Dataset
class MultiviewData(Dataset):
    # 初始化函数，接受数据库名称、设备和路径作为参数
    def __init__(self, db, device, path="datasets/"):
        self.data_views = list()  # 初始化数据视图列表

        # 处理MSRCv1数据集
        if db == "MSRCv1":
            mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))  # 加载MATLAB文件
            X_data = mat['X']  # 获取数据
            print(X_data.shape)
            print(X_data[0].shape)
            self.num_views = X_data.shape[1]  # 获取视图数量
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))  # 添加每个视图的数据，并转换为float32类型
            scaler = MinMaxScaler()  # 创建MinMaxScaler实例
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])  # 对数据进行归一化
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理MNIST-USPS数据集
        elif db == "MNIST-USPS":
            mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))  # 加载MATLAB文件
            X1 = mat['X1'].astype(np.float32)  # 获取并转换第一个视图的数据
            X2 = mat['X2'].astype(np.float32)  # 获取并转换第二个视图的数据
            self.data_views.append(X1.reshape(X1.shape[0], -1))  # 将第一个视图的数据展平并添加到数据视图列表中
            self.data_views.append(X2.reshape(X2.shape[0], -1))  # 将第二个视图的数据展平并添加到数据视图列表中
            self.num_views = len(self.data_views)  # 获取视图数量
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理BDGP数据集
        elif db == "BDGP":
            mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))  # 加载MATLAB文件
            X1 = mat['X1'].astype(np.float32)  # 获取并转换第一个视图的数据
            X2 = mat['X2'].astype(np.float32)  # 获取并转换第二个视图的数据
            self.data_views.append(X1)  # 添加第一个视图的数据
            self.data_views.append(X2)  # 添加第二个视图的数据
            self.num_views = len(self.data_views)  # 获取视图数量
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理Fashion数据集
        elif db == "Fashion":
            mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))  # 加载MATLAB文件
            X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)  # 将第一个视图的数据展平并转换为float32类型
            X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)  # 将第二个视图的数据展平并转换为float32类型
            X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)  # 将第三个视图的数据展平并转换为float32类型
            self.data_views.append(X1)  # 添加第一个视图的数据
            self.data_views.append(X2)  # 添加第二个视图的数据
            self.data_views.append(X3)  # 添加第三个视图的数据
            self.num_views = len(self.data_views)  # 获取视图数量
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理COIL20数据集
        elif db == "COIL20":
            mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))  # 加载MATLAB文件
            X_data = mat['X']  # 获取数据
            self.num_views = X_data.shape[1]  # 获取视图数量
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))  # 添加每个视图的数据，并转换为float32类型
            scaler = MinMaxScaler()  # 创建MinMaxScaler实例
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])  # 对数据进行归一化
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理手写数据集
        elif db == "hand":
            mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))  # 加载MATLAB文件
            X_data = mat['X']  # 获取数据
            self.num_views = X_data.shape[1]  # 获取视图数量
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))  # 添加每个视图的数据，并转换为float32类型
            scaler = MinMaxScaler()  # 创建MinMaxScaler实例
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])  # 对数据进行归一化
            self.labels = np.array(np.squeeze(mat['Y']) + 1).astype(np.int32)  # 获取标签，并转换为int32类型

        # 处理Scene数据集
        elif db == "scene":
            mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))  # 加载MATLAB文件
            X_data = mat['X']  # 获取数据
            self.num_views = X_data.shape[1]  # 获取视图数量
            for idx in range(self.num_views):
                self.data_views.append(X_data[0, idx].astype(np.float32))  # 添加每个视图的数据，并转换为float32类型
            scaler = MinMaxScaler()  # 创建MinMaxScaler实例
            for idx in range(self.num_views):
                self.data_views[idx] = scaler.fit_transform(self.data_views[idx])  # 对数据进行归一化
            self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)  # 获取标签，并转换为int32类型

        # 如果数据库名称不匹配任何已知的数据集，则抛出未实现错误
        else:
            raise NotImplementedError

        # 将数据视图转换为PyTorch张量，并将其移动到指定设备
        for idx in range(self.num_views):
            self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)

    # 获取数据集的长度（样本数量）
    def __len__(self):
        return len(self.labels)

    # 获取指定索引的数据和标签
    def __getitem__(self, index):
        sub_data_views = list()  # 初始化子数据视图列表
        for view_idx in range(self.num_views):
            data_view = self.data_views[view_idx]  # 获取视图数据
            sub_data_views.append(data_view[index])  # 添加指定索引的数据到子数据视图列表

        return sub_data_views, self.labels[index]  # 返回子数据视图和标签


# 获取多视图数据的批量加载器
def get_multiview_data(mv_data, batch_size):
    num_views = len(mv_data.data_views)  # 获取视图数量
    num_samples = len(mv_data.labels)  # 获取样本数量
    num_clusters = len(np.unique(mv_data.labels))  # 获取聚类数量

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=batch_size,
        shuffle=True,  # 打乱数据
        drop_last=True,  # 丢弃最后一个不足批量大小的批次
    )

    return mv_data_loader, num_views, num_samples, num_clusters  # 返回数据加载器，视图数量，样本数量和聚类数量


# 获取所有多视图数据的批量加载器
def get_all_multiview_data(mv_data):
    num_views = len(mv_data.data_views)  # 获取视图数量
    num_samples = len(mv_data.labels)  # 获取样本数量
    num_clusters = len(np.unique(mv_data.labels))  # 获取聚类数量

    mv_data_loader = torch.utils.data.DataLoader(
        mv_data,
        batch_size=num_samples,  # 设置批量大小为样本数量
        shuffle=True,  # 打乱数据
        drop_last=True,  # 丢弃最后一个不足批量大小的批次
    )

    return mv_data_loader, num_views, num_samples, num_clusters  # 返回数据加载器，视图数量，样本数量和聚类数量


if __name__ == "__main__":
    mul = MultiviewData("MSRCv1",'cuda')

# import os, random, sys
#
# import torch
# import numpy as np
# import scipy.io as sio
# from sklearn.preprocessing import MinMaxScaler
#
# from torch.utils.data import Dataset
# # from torch.nn.functional import normalize
# from utils import *
#
#
# class MultiviewData(Dataset):
#     def __init__(self, db, device, path="datasets/"):
#         self.data_views = list()
#
#         if db == "MSRCv1":
#             mat = sio.loadmat(os.path.join(path, 'MSRCv1.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "MNIST-USPS":
#             mat = sio.loadmat(os.path.join(path, 'MNIST_USPS.mat'))
#             X1 = mat['X1'].astype(np.float32)
#             X2 = mat['X2'].astype(np.float32)
#             self.data_views.append(X1.reshape(X1.shape[0], -1))
#             self.data_views.append(X2.reshape(X2.shape[0], -1))
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "BDGP":
#             mat = sio.loadmat(os.path.join(path, 'BDGP.mat'))
#             X1 = mat['X1'].astype(np.float32)
#             X2 = mat['X2'].astype(np.float32)
#             self.data_views.append(X1)
#             self.data_views.append(X2)
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "Fashion":
#             mat = sio.loadmat(os.path.join(path, 'Fashion.mat'))
#             X1 = mat['X1'].reshape(mat['X1'].shape[0], mat['X1'].shape[1] * mat['X1'].shape[2]).astype(np.float32)
#             X2 = mat['X2'].reshape(mat['X2'].shape[0], mat['X2'].shape[1] * mat['X2'].shape[2]).astype(np.float32)
#             X3 = mat['X3'].reshape(mat['X3'].shape[0], mat['X3'].shape[1] * mat['X3'].shape[2]).astype(np.float32)
#             self.data_views.append(X1)
#             self.data_views.append(X2)
#             self.data_views.append(X3)
#             self.num_views = len(self.data_views)
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "COIL20":
#             mat = sio.loadmat(os.path.join(path, 'COIL20.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         elif db == "hand":
#             mat = sio.loadmat(os.path.join(path, 'handwritten.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])+1).astype(np.int32)
#
#         elif db == "scene":
#             mat = sio.loadmat(os.path.join(path, 'Scene15.mat'))
#             X_data = mat['X']
#             self.num_views = X_data.shape[1]
#             for idx in range(self.num_views):
#                 self.data_views.append(X_data[0, idx].astype(np.float32))
#             scaler = MinMaxScaler()
#             for idx in range(self.num_views):
#                 self.data_views[idx] = scaler.fit_transform(self.data_views[idx])
#             self.labels = np.array(np.squeeze(mat['Y'])).astype(np.int32)
#
#         else:
#             raise NotImplementedError
#
#         for idx in range(self.num_views):
#             self.data_views[idx] = torch.from_numpy(self.data_views[idx]).to(device)
#
#     def __len__(self):
#         return len(self.labels)
#
#     def __getitem__(self, index):
#         sub_data_views = list()
#         for view_idx in range(self.num_views):
#             data_view = self.data_views[view_idx]
#             sub_data_views.append(data_view[index])
#
#         return sub_data_views, self.labels[index]
#
#
# def get_multiview_data(mv_data, batch_size):
#     num_views = len(mv_data.data_views)
#     num_samples = len(mv_data.labels)
#     num_clusters = len(np.unique(mv_data.labels))
#
#     mv_data_loader = torch.utils.data.DataLoader(
#         mv_data,
#         batch_size=batch_size,
#         shuffle=True,
#         drop_last=True,
#     )
#
#     return mv_data_loader, num_views, num_samples, num_clusters
#
#
# def get_all_multiview_data(mv_data):
#     num_views = len(mv_data.data_views)
#     num_samples = len(mv_data.labels)
#     num_clusters = len(np.unique(mv_data.labels))
#
#     mv_data_loader = torch.utils.data.DataLoader(
#         mv_data,
#         batch_size=num_samples,
#         shuffle=True,
#         drop_last=True,
#     )
#
#     return mv_data_loader, num_views, num_samples, num_clusters
