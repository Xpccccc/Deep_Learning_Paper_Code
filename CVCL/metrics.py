import numpy as np  # 导入NumPy库
from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
from scipy.optimize import linear_sum_assignment  # 从SciPy库中导入线性求解分配算法


def calculate_metrics(label, pred):
    """
    计算聚类评估指标。

    :param label: 真实标签
    :param pred: 预测标签
    :return: 准确率（ACC），归一化互信息（NMI），纯度（PUR），调整兰德指数（ARI）
    """
    acc = calculate_acc(label, pred)  # 计算准确率
    nmi = normalized_mutual_info_score(label, pred)  # 计算归一化互信息
    pur = calculate_purity(label, pred)  # 计算纯度
    ari = adjusted_rand_score(label, pred)  # 计算调整兰德指数

    return acc, nmi, pur, ari  # 返回所有指标


def calculate_acc(y_true, y_pred):
    """
    计算聚类准确率。

    :param y_true: 真实标签，numpy数组，形状为 `(n_samples,)`
    :param y_pred: 预测标签，numpy数组，形状为 `(n_samples,)`
    :return: 准确率，范围在 [0, 1] 之间
    """
    y_true = y_true.astype(np.int64)  # 将真实标签转换为整数类型
    assert y_pred.size == y_true.size  # 确保预测标签和真实标签的大小相同
    D = max(y_pred.max(), y_true.max()) + 1  # 计算标签的最大值 + 1
    w = np.zeros((D, D), dtype=np.int64)  # 初始化标签匹配矩阵

    for i in range(y_pred.size):  # 遍历每个样本
        w[y_pred[i], y_true[i]] += 1  # 更新标签匹配矩阵

    ind_row, ind_col = linear_sum_assignment(w.max() - w)  # 使用匈牙利算法计算最佳标签匹配

    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size  # 计算准确率


def calculate_purity(y_true, y_pred):
    """
    计算聚类纯度。

    :param y_true: 真实标签，numpy数组，形状为 `(n_samples,)`
    :param y_pred: 预测标签，numpy数组，形状为 `(n_samples,)`
    :return: 纯度，范围在 [0, 1] 之间
    """
    y_voted_labels = np.zeros(y_true.shape)  # 初始化投票后的标签数组
    labels = np.unique(y_true)  # 获取所有唯一的真实标签
    ordered_labels = np.arange(labels.shape[0])  # 创建标签的顺序数组
    for k in range(labels.shape[0]):  # 将真实标签映射到顺序标签
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)  # 重新获取唯一的真实标签
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)  # 创建用于直方图的边界

    for cluster_index in np.unique(y_pred):  # 遍历每个聚类
        hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)  # 计算直方图
        winner = np.argmax(hist)  # 获取最频繁的标签
        y_voted_labels[y_pred == cluster_index] = winner  # 为每个聚类分配最频繁的标签

    return accuracy_score(y_true, y_voted_labels)  # 计算纯度并返回

# import numpy as np
# from sklearn.metrics import normalized_mutual_info_score, v_measure_score, adjusted_rand_score, accuracy_score
# from scipy.optimize import linear_sum_assignment
#
#
# def calculate_metrics(label, pred):
#     acc = calculate_acc(label, pred)
#     # nmi = v_measure_score(label, pred)
#     nmi = normalized_mutual_info_score(label, pred)
#     pur = calculate_purity(label, pred)
#     ari = adjusted_rand_score(label, pred)
#
#     return acc, nmi, pur, ari
#
#
# def calculate_acc(y_true, y_pred):
#     """
#     Calculate clustering accuracy.
#     # Arguments
#         y: true labels, numpy.array with shape `(n_samples,)`
#         y_pred: predicted labels, numpy.array with shape `(n_samples,)`
#     # Return
#         accuracy, in [0,1]
#     """
#     y_true = y_true.astype(np.int64)
#     assert y_pred.size == y_true.size
#     D = max(y_pred.max(), y_true.max()) + 1
#     w = np.zeros((D, D), dtype=np.int64)
#     for i in range(y_pred.size):
#         w[y_pred[i], y_true[i]] += 1
#
#     ind_row, ind_col = linear_sum_assignment(w.max() - w)
#
#     # u = linear_sum_assignment(w.max() - w)
#     # ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
#     # return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
#
#     return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size
#
#
# def calculate_purity(y_true, y_pred):
#     y_voted_labels = np.zeros(y_true.shape)
#     labels = np.unique(y_true)
#     ordered_labels = np.arange(labels.shape[0])
#     for k in range(labels.shape[0]):
#         y_true[y_true == labels[k]] = ordered_labels[k]
#     labels = np.unique(y_true)
#     bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)
#
#     for cluster_index in np.unique(y_pred):
#         hist, _ = np.histogram(y_true[y_pred == cluster_index], bins=bins)
#         winner = np.argmax(hist)
#         y_voted_labels[y_pred == cluster_index] = winner
#
#     return accuracy_score(y_true, y_voted_labels)