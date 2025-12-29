import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix


# TODO 1.计算聚类的准确率acc
def cluster_accuracy(true_labels, pred_labels):
    # 创建一个混淆矩阵
    n_clusters = max(pred_labels.max(), true_labels.max()) + 1  # 获取最大的标签数
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(true_labels.size):
        confusion_matrix[true_labels[i], pred_labels[i]] += 1
    # 使用匈牙利算法（线性分配问题）来寻找最优标签重映射
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)  # 我们求最大匹配所以用负号
    # 计算最终地聚类准确率
    return confusion_matrix[row_ind, col_ind].sum() / true_labels.size


# TODO 2.计算聚类的归一化互信息（NMI）
def cluster_nmi(true_labels, pred_labels):
    assert len(true_labels) == len(pred_labels), "标签数量不匹配"
    n = len(true_labels)
    true_label_counts = np.bincount(true_labels)
    pred_label_counts = np.bincount(pred_labels)
    true_label_probs = true_label_counts / n
    pred_label_probs = pred_label_counts / n
    # 添加平滑值
    eps = 1e-9
    true_label_probs_smooth = true_label_probs + eps
    pred_label_probs_smooth = pred_label_probs + eps
    # 计算互信息
    mi = 0
    for i in range(len(true_label_probs)):
        for j in range(len(pred_label_probs)):
            pij = np.sum(np.logical_and(true_labels == i, pred_labels == j)) / n
            if pij > 0:
                mi += pij * np.log((pij / (true_label_probs_smooth[i] * pred_label_probs_smooth[j])))
    # 计算归一化互信息
    h_true = -np.sum(true_label_probs_smooth * np.log(true_label_probs_smooth))
    h_pred = -np.sum(pred_label_probs_smooth * np.log(pred_label_probs_smooth))
    # 修正分母部分，确保其值始终为正数
    denom = np.sqrt(np.abs(h_true * h_pred))
    nmi = mi / max(denom, eps)
    return nmi


# TODO 3.计算聚类的调整兰德指数（ARI）
def cluster_ari(true_labels, pred_labels):
    assert len(true_labels) == len(pred_labels), "标签数量不匹配"
    # 确保真实标签和预测标签的类别数量一致
    num_classes = max(max(true_labels), max(pred_labels)) + 1
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    # 计算共现矩阵
    contingency_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for i in range(len(true_labels)):
        contingency_matrix[true_labels[i], pred_labels[i]] += 1
    # 计算各种组合的数量
    a = np.sum(contingency_matrix, axis=1)
    b = np.sum(contingency_matrix, axis=0)
    ab = np.sum(np.multiply(contingency_matrix, contingency_matrix))
    # 计算调整兰德指数
    index = (ab - np.sum(a * b) / len(true_labels)) / (
            0.5 * (np.sum(a ** 2) + np.sum(b ** 2)) - np.sum(a * b) / len(true_labels))
    return index


# TODO 4.计算聚类的纯度 (Purity)
def cluster_purity(true_labels, pred_labels):
    assert len(true_labels) == len(pred_labels), "标签数量不匹配"
    n = len(true_labels)
    num_clusters = np.unique(pred_labels).shape[0]
    purity = 0
    for cluster in range(num_clusters):
        cluster_indices = pred_labels == cluster
        if np.any(cluster_indices):
            majority_label = np.argmax(np.bincount(true_labels[cluster_indices]))
            purity += np.sum(cluster_indices & (true_labels == majority_label))
    purity /= n
    return purity


# TODO 5. 计算聚类的F分数(Fscore)
def cluster_Fscore(true_labels, pred_labels):
    precision = cluster_precision(true_labels, pred_labels)
    recall = cluster_recall(true_labels, pred_labels)
    # TODO F分数(Fscore)计算公式
    fscore = 2 * precision * recall / (precision + recall)
    return fscore


# TODO 6.1计算召回率(recall)
def cluster_recall(true_labels, pred_labels):
    # 计算混淆矩阵
    num_classes = max(np.max(true_labels), np.max(pred_labels)) + 1
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true_label, pred_label in zip(true_labels, pred_labels):
        confusion_matrix[true_label, pred_label] += 1
    # 使用匈牙利算法寻找最优标签重映射
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)
    # 计算总召回率
    total_recall = confusion_matrix[row_ind, col_ind].sum() / len(true_labels)
    return total_recall


# TODO 6.2 precision
def cluster_precision(true_labels, predicted_labels):
    """
    计算聚类的查准率（Precision）。

    参数：
    true_labels: 实际标签的列表或数组
    predicted_labels: 聚类算法生成的预测标签的列表或数组

    返回：
    每个簇的查准率和总体查准率
    """
    # 计算混淆矩阵
    contingency = contingency_matrix(true_labels, predicted_labels)

    # 计算每个簇的查准率
    precisions = np.zeros(contingency.shape[1])
    for j in range(contingency.shape[1]):
        if np.sum(contingency[:, j]) != 0:
            precisions[j] = contingency[j, j] / np.sum(contingency[:, j])

    # 总体查准率
    overall_precision = np.mean(precisions)

    return overall_precision


# TODO 评价指标
def compute_metric(Y_ndarray, Y_pre):
    # Y_ndarray是真实的标签数组，Y_pre是预测的标签结果[通常需要经过argmax处理再输出]
    Y_ndarray = Y_ndarray.astype(np.int64)
    Y_pre = Y_pre.astype(np.int64)
    # TODO 1.计算准确率 (ACC)
    acc_cluster = cluster_accuracy(Y_ndarray, Y_pre)
    # print("\n1.[ACC_cluster.py]:{:.5f}".format(acc_cluster))

    # TODO 2.计算归一化互信息 (NMI)
    nmi_cluster = cluster_nmi(Y_ndarray, Y_pre)
    # print("2.[NMI_cluster.py]:{:.5f}".format(nmi_cluster))

    # TODO 3.计算调整兰德指数 (ARI)
    ari_cluster = cluster_ari(Y_ndarray, Y_pre)
    # print("3.[ARI_cluster.py]:{:.5f}".format(ari_cluster))

    # TODO 4.计算纯度 (Purity)
    pur_cluster = cluster_purity(Y_ndarray, Y_pre)
    # print("4.[PUR_cluster.py]:{:.5f}".format(pur_cluster))

    # TODO 5. 计算F分数(Fscore)
    fscore_cluster = cluster_Fscore(Y_ndarray, Y_pre)
    # print("5.[Fscore_cluster.py]:{:.5f}".format(fscore_cluster))

    # TODO 6.1 recall
    recall_cluster = cluster_recall(Y_ndarray, Y_pre)
    # TODO 6.2 precision
    precision_cluster = cluster_precision(Y_ndarray, Y_pre)

    return float(acc_cluster), float(nmi_cluster), float(pur_cluster), float(ari_cluster), float(fscore_cluster), float(precision_cluster), float(recall_cluster)