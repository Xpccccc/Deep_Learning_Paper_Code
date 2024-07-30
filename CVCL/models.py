import time  # 导入时间模块，用于计算运行时间

import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch中的神经网络模块
from loss import *  # 从loss模块中导入所有内容
from metrics import *  # 从metrics模块中导入所有内容
from dataprocessing import *  # 从dataprocessing模块中导入所有内容


def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
    """
    对网络模型进行预训练。

    :param network_model: 网络模型
    :param mv_data: 多视角数据
    :param batch_size: 批次大小
    :param epochs: 训练轮数
    :param optimizer: 优化器
    :return: 预训练过程中的损失值
    """
    t = time.time()  # 记录开始时间
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)  # 获取数据加载器和相关信息

    pre_train_loss_values = np.zeros(epochs, dtype=np.float64)  # 初始化损失值数组

    criterion = torch.nn.MSELoss()  # 定义均方误差损失函数
    for epoch in range(epochs):  # 进行每一轮训练
        total_loss = 0.  # 初始化总损失
        for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):  # 遍历每一个批次
            _, dvs, _ = network_model(sub_data_views)  # 获取模型输出
            loss_list = list()  # 初始化损失值列表
            for idx in range(num_views):  # 对每个视角计算损失
                loss_list.append(criterion(sub_data_views[idx], dvs[idx]))  # 计算每个视角的均方误差损失
            loss = sum(loss_list)  # 计算总损失
            optimizer.zero_grad()  # 清零梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()  # 累加总损失

        pre_train_loss_values[epoch] = total_loss  # 保存每轮的损失值
        if epoch % 10 == 0 or epoch == epochs - 1:  # 每10轮或者最后一轮打印损失
            print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))

    print("Pre-training finished.")  # 打印预训练完成的消息
    print("Total time elapsed: {:.4f}s".format(time.time() - t))  # 打印总耗时

    return pre_train_loss_values  # 返回预训练损失值


def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta, temperature_l, normalized, epoch,
                      optimizer):
    """
    对网络模型进行对比训练。

    :param network_model: 网络模型
    :param mv_data: 多视角数据
    :param mvc_loss: 对比损失函数
    :param batch_size: 批次大小
    :param lmd: 损失函数的权重参数
    :param beta: 损失函数的权重参数
    :param temperature_l: 温度参数
    :param normalized: 是否归一化
    :param epoch: 当前轮数
    :param optimizer: 优化器
    :return: 总损失
    """
    network_model.train()  # 设置模型为训练模式
    mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)  # 获取数据加载器和相关信息
    criterion = torch.nn.MSELoss()  # 定义均方误差损失函数
    total_loss = 0.  # 初始化总损失
    for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):  # 遍历每一个批次
        lbps, dvs, _ = network_model(sub_data_views)  # 获取模型输出
        loss_list = list()  # 初始化损失值列表
        for i in range(num_views):  # 对每对视角计算对比损失
            for j in range(i + 1, num_views):
                loss_list.append(lmd * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))  # 标签损失
                loss_list.append(beta * mvc_loss.forward_prob(lbps[i], lbps[j]))  # 概率损失
            loss_list.append(criterion(sub_data_views[i], dvs[i]))  # 重建损失
        loss = sum(loss_list)  # 计算总损失
        optimizer.zero_grad()  # 清零梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        total_loss += loss.item()  # 累加总损失

    if epoch % 10 == 0:  # 每10轮打印损失
        print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))

    return total_loss  # 返回总损失


def inference(network_model, mv_data, batch_size):
    """
    对网络模型进行推断。

    :param network_model: 网络模型
    :param mv_data: 多视角数据
    :param batch_size: 批次大小
    :return: 总预测标签、每个视角的预测向量和真实标签向量
    """
    network_model.eval()  # 设置模型为评估模式
    mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)  # 获取数据加载器和相关信息

    soft_vector = []  # 存储软标签向量
    pred_vectors = []  # 存储每个视角的预测向量
    labels_vector = []  # 存储真实标签
    for v in range(num_views):  # 初始化每个视角的预测向量
        pred_vectors.append([])

    for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):  # 遍历每一个批次
        with torch.no_grad():  # 禁用梯度计算
            lbps, _, _ = network_model(sub_data_views)  # 获取模型输出
            lbp = sum(lbps) / num_views  # 计算平均标签概率

        for idx in range(num_views):  # 对每个视角进行处理
            pred_label = torch.argmax(lbps[idx], dim=1)  # 获取预测标签
            pred_vectors[idx].extend(pred_label.detach().cpu().numpy())  # 保存预测标签

        soft_vector.extend(lbp.detach().cpu().numpy())  # 保存软标签向量
        labels_vector.extend(sub_labels)  # 保存真实标签

    for idx in range(num_views):  # 转换预测向量为numpy数组
        pred_vectors[idx] = np.array(pred_vectors[idx])

    actual_num_samples = len(soft_vector)  # 获取样本数量
    labels_vector = np.array(labels_vector).reshape(actual_num_samples)  # 转换真实标签为numpy数组
    total_pred = np.argmax(np.array(soft_vector), axis=1)  # 获取最终预测标签

    return total_pred, pred_vectors, labels_vector  # 返回预测标签、每个视角的预测向量和真实标签


def valid(network_model, mv_data, batch_size):
    """
    验证模型的效果。

    :param network_model: 网络模型
    :param mv_data: 多视角数据
    :param batch_size: 批次大小
    :return: 评估指标（ACC, NMI, PUR, ARI）
    """
    total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)  # 获取推断结果
    num_views = len(mv_data.data_views)  # 获取视角数量

    print("Clustering results on cluster assignments of each view:")  # 打印每个视角的聚类结果
    for idx in range(num_views):  # 对每个视角计算评估指标
        acc, nmi, pur, ari = calculate_metrics(labels_vector, pred_vectors[idx])  # 计算评估指标
        print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx + 1, acc,
                                                                                 idx + 1, nmi,
                                                                                 idx + 1, pur,
                                                                                 idx + 1, ari))

    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))  # 打印基于语义标签的聚类结果
    acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)  # 计算评估指标
    print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))

    return acc, nmi, pur, ari  # 返回评估指标

# import time
#
# import torch
# import torch.nn as nn
# from loss import *
# from metrics import *
# from dataprocessing import *
#
#
# def pre_train(network_model, mv_data, batch_size, epochs, optimizer):
#     t = time.time()
#     mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
#
#     pre_train_loss_values = np.zeros(epochs, dtype=np.float64)
#
#     criterion = torch.nn.MSELoss()
#     for epoch in range(epochs):
#         total_loss = 0.
#         for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
#             _, dvs, _ = network_model(sub_data_views)
#             loss_list = list()
#             for idx in range(num_views):
#                 loss_list.append(criterion(sub_data_views[idx], dvs[idx]))
#             loss = sum(loss_list)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#
#         pre_train_loss_values[epoch] = total_loss
#         if epoch % 10 == 0 or epoch == epochs - 1:
#             print('Pre-training, epoch {}, Loss:{:.7f}'.format(epoch, total_loss / num_samples))
#
#     print("Pre-training finished.")
#     print("Total time elapsed: {:.4f}s".format(time.time() - t))
#
#     return pre_train_loss_values
#
#
# def contrastive_train(network_model, mv_data, mvc_loss, batch_size, lmd, beta, temperature_l, normalized, epoch,
#                       optimizer):
#
#     network_model.train()
#     mv_data_loader, num_views, num_samples, num_clusters = get_multiview_data(mv_data, batch_size)
#     criterion = torch.nn.MSELoss()
#     total_loss = 0.
#     for batch_idx, (sub_data_views, _) in enumerate(mv_data_loader):
#         lbps, dvs, _ = network_model(sub_data_views)
#         loss_list = list()
#         for i in range(num_views):
#             for j in range(i + 1, num_views):
#                 loss_list.append(lmd * mvc_loss.forward_label(lbps[i], lbps[j], temperature_l, normalized))
#                 loss_list.append(beta * mvc_loss.forward_prob(lbps[i], lbps[j]))
#             loss_list.append(criterion(sub_data_views[i], dvs[i]))
#         loss = sum(loss_list)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#
#     if epoch % 10 == 0:
#         print('Contrastive_train, epoch {} loss:{:.7f}'.format(epoch, total_loss / num_samples))
#
#     return total_loss
#
#
# def inference(network_model, mv_data, batch_size):
#
#     network_model.eval()
#     mv_data_loader, num_views, num_samples, _ = get_multiview_data(mv_data, batch_size)
#
#     soft_vector = []
#     pred_vectors = []
#     labels_vector = []
#     for v in range(num_views):
#         pred_vectors.append([])
#
#     for batch_idx, (sub_data_views, sub_labels) in enumerate(mv_data_loader):
#         with torch.no_grad():
#             lbps, _, _ = network_model(sub_data_views)
#             lbp = sum(lbps)/num_views
#
#         for idx in range(num_views):
#             pred_label = torch.argmax(lbps[idx], dim=1)
#             pred_vectors[idx].extend(pred_label.detach().cpu().numpy())
#
#         soft_vector.extend(lbp.detach().cpu().numpy())
#         labels_vector.extend(sub_labels)
#
#     for idx in range(num_views):
#         pred_vectors[idx] = np.array(pred_vectors[idx])
#
#     # labels_vector = np.array(labels_vector).reshape(num_samples)
#     actual_num_samples = len(soft_vector)
#     labels_vector = np.array(labels_vector).reshape(actual_num_samples)
#     total_pred = np.argmax(np.array(soft_vector), axis=1)
#
#     return total_pred, pred_vectors, labels_vector
#
#
# def valid(network_model, mv_data, batch_size):
#
#     total_pred, pred_vectors, labels_vector = inference(network_model, mv_data, batch_size)
#     num_views = len(mv_data.data_views)
#
#     print("Clustering results on cluster assignments of each view:")
#     for idx in range(num_views):
#         acc, nmi, pur, ari = calculate_metrics(labels_vector,  pred_vectors[idx])
#         print('ACC{} = {:.4f} NMI{} = {:.4f} PUR{} = {:.4f} ARI{}={:.4f}'.format(idx+1, acc,
#                                                                                  idx+1, nmi,
#                                                                                  idx+1, pur,
#                                                                                  idx+1, ari))
#
#     print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
#     acc, nmi, pur, ari = calculate_metrics(labels_vector, total_pred)
#     print('ACC = {:.4f} NMI = {:.4f} PUR = {:.4f} ARI={:.4f}'.format(acc, nmi, pur, ari))
#
#     return acc, nmi, pur, ari
