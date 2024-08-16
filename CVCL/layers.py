import torch.nn as nn  # 导入PyTorch中的神经网络模块


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        """
        初始化AutoEncoder类。

        :param input_dim: 输入数据的维度
        :param feature_dim: 编码后的特征维度
        :param dims: 隐藏层的维度列表
        """
        super(AutoEncoder, self).__init__()  # 调用父类的初始化方法
        self.encoder = nn.Sequential()  # 创建一个顺序容器用于构建编码器
        for i in range(len(dims) + 1):  # 遍历每一层
            if i == 0:
                # 第一层，输入层到第一个隐藏层
                self.encoder.add_module('Linear%d' % i, nn.Linear(input_dim, dims[i]))
            elif i == len(dims):
                # 最后一层，最后一个隐藏层到特征层
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], feature_dim))
            else:
                # 隐藏层之间的线性变换
                self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            # 添加ReLU激活函数
            self.encoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入数据
        :return: 编码后的数据
        """
        return self.encoder(x)  # 将输入数据通过编码器进行处理


class AutoDecoder(nn.Module):
    def __init__(self, input_dim, feature_dim, dims):
        """
        初始化AutoDecoder类。

        :param input_dim: 输入数据的维度
        :param feature_dim: 编码后的特征维度
        :param dims: 隐藏层的维度列表
        """
        super(AutoDecoder, self).__init__()  # 调用父类的初始化方法
        self.decoder = nn.Sequential()  # 创建一个顺序容器用于构建解码器
        dims = list(reversed(dims))  # 隐藏层维度反转，用于解码器
        for i in range(len(dims) + 1):  # 遍历每一层
            if i == 0:
                # 第一层，特征层到第一个隐藏层
                self.decoder.add_module('Linear%d' % i, nn.Linear(feature_dim, dims[i]))
            elif i == len(dims):
                # 最后一层，最后一个隐藏层到输出层
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], input_dim))
            else:
                # 隐藏层之间的线性变换
                self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i - 1], dims[i]))
            # 添加ReLU激活函数
            self.decoder.add_module('relu%d' % i, nn.ReLU())

    def forward(self, x):
        """
        前向传播函数。

        :param x: 输入数据
        :return: 解码后的数据
        """
        return self.decoder(x)  # 将输入数据通过解码器进行处理


class CVCLNetwork(nn.Module):
    def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
        """
        初始化CVCLNetwork类。

        :param num_views: 数据视角的数量
        :param input_sizes: 每个视角的数据维度
        :param dims: 隐藏层的维度列表
        :param dim_high_feature: 编码后的高维特征维度
        :param dim_low_feature: 标签学习模块中的低维特征维度
        :param num_clusters: 聚类的数量
        """
        super(CVCLNetwork, self).__init__()  # 调用父类的初始化方法
        self.encoders = list()  # 创建一个列表用于存储多个编码器
        self.decoders = list()  # 创建一个列表用于存储多个解码器
        for idx in range(num_views):
            # 为每个视角创建对应的编码器和解码器
            self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
            self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
        self.encoders = nn.ModuleList(self.encoders)  # 将编码器列表转换为ModuleList
        self.decoders = nn.ModuleList(self.decoders)  # 将解码器列表转换为ModuleList

        # 标签学习模块，用于将高维特征映射到簇标签
        self.label_learning_module = nn.Sequential(
            nn.Linear(dim_high_feature, dim_low_feature),  # 高维特征到低维特征
            nn.Linear(dim_low_feature, num_clusters),  # 低维特征到簇标签
            nn.Softmax(dim=1)  # Softmax激活函数，输出概率分布
        )

    def forward(self, data_views):
        """
        前向传播函数。

        :param data_views: 输入的多个视角数据
        :return: 标签概率、解码后的数据和高维特征
        """
        lbps = list()  # 存储每个视角的标签概率
        dvs = list()  # 存储每个视角的解码数据
        features = list()  # 存储每个视角的高维特征

        num_views = len(data_views)  # 获取视角的数量
        for idx in range(num_views):
            data_view = data_views[idx]  # 获取当前视角的数据
            high_features = self.encoders[idx](data_view)  # 通过编码器获取高维特征
            label_probs = self.label_learning_module(high_features)  # 通过标签学习模块获取标签概率
            data_view_recon = self.decoders[idx](high_features)  # 通过解码器获取解码后的数据
            features.append(high_features)  # 保存高维特征
            lbps.append(label_probs)  # 保存标签概率
            dvs.append(data_view_recon)  # 保存解码数据

        return lbps, dvs, features  # 返回标签概率、解码后的数据和高维特征


# import torch.nn as nn
#
#
# class AutoEncoder(nn.Module):
#     def __init__(self, input_dim, feature_dim, dims):
#         super(AutoEncoder, self).__init__()
#         self.encoder = nn.Sequential()
#         for i in range(len(dims)+1):
#             if i == 0:
#                 self.encoder.add_module('Linear%d' % i,  nn.Linear(input_dim, dims[i]))
#             elif i == len(dims):
#                 self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], feature_dim))
#             else:
#                 self.encoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
#             self.encoder.add_module('relu%d' % i, nn.ReLU())
#
#     def forward(self, x):
#         return self.encoder(x)
#
#
# class AutoDecoder(nn.Module):
#     def __init__(self, input_dim, feature_dim, dims):
#         super(AutoDecoder, self).__init__()
#         self.decoder = nn.Sequential()
#         dims = list(reversed(dims))
#         for i in range(len(dims)+1):
#             if i == 0:
#                 self.decoder.add_module('Linear%d' % i,  nn.Linear(feature_dim, dims[i]))
#             elif i == len(dims):
#                 self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], input_dim))
#             else:
#                 self.decoder.add_module('Linear%d' % i, nn.Linear(dims[i-1], dims[i]))
#             self.decoder.add_module('relu%d' % i, nn.ReLU())
#
#     def forward(self, x):
#         return self.decoder(x)
#
#
# class CVCLNetwork(nn.Module):
#     def __init__(self, num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters):
#         super(CVCLNetwork, self).__init__()
#         self.encoders = list()
#         self.decoders = list()
#         for idx in range(num_views):
#             self.encoders.append(AutoEncoder(input_sizes[idx], dim_high_feature, dims))
#             self.decoders.append(AutoDecoder(input_sizes[idx], dim_high_feature, dims))
#         self.encoders = nn.ModuleList(self.encoders)
#         self.decoders = nn.ModuleList(self.decoders)
#
#         self.label_learning_module = nn.Sequential(
#             nn.Linear(dim_high_feature, dim_low_feature),
#             nn.Linear(dim_low_feature, num_clusters),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, data_views):
#         lbps = list()
#         dvs = list()
#         features = list()
#
#         num_views = len(data_views)
#         for idx in range(num_views):
#             data_view = data_views[idx]
#             high_features = self.encoders[idx](data_view)
#             label_probs = self.label_learning_module(high_features)
#             data_view_recon = self.decoders[idx](high_features)
#             features.append(high_features)
#             lbps.append(label_probs)
#             dvs.append(data_view_recon)
#
#         return lbps, dvs, features
