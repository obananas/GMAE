import torch


def contrastive_loss(args, hidden, nbr_idx, neg_idx, idx):
    if not args.do_contrast:
        return 0
    loss_con = 0
    # 遍历每个样本，计算正负样本对的对比损失
    for i in range(len(idx)):
        # 获取正样本表示，nbr_idx[i]是正样本的索引
        hidden_positive = hidden[nbr_idx[i]]
        # 计算正样本和当前样本的余弦相似度，并取其指数值
        positive = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_positive.detach()))
        # 获取负样本表示，neg_idx[i]是负样本的索引
        hidden_negative = hidden[neg_idx[i]]
        # 计算负样本和当前样本的余弦相似度，并取其指数值
        negative = torch.exp(torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_negative.detach())).sum()
        # 计算对比损失：对比正样本和负样本的相似度，最大化正样本与负样本的差异
        loss_con -= torch.log((positive / negative)).sum()
        # 清除 GPU 缓存，避免内存溢出
        torch.cuda.empty_cache()
    # 返回平均对比损失（取决于样本数量 idx 的长度）
    return loss_con / len(idx)


# 正交损失函数（Orthogonal Loss）用于使共享潜在表示与每个视图的特定潜在表示之间的相关性最小化
def orthogonal_loss(shared, specific):
    # 计算共享潜在表示的均值并减去均值
    _shared = shared.detach()
    _shared = _shared - _shared.mean(dim=0)
    # 计算共享表示和特定表示之间的相关性矩阵
    correlation_matrix = _shared.t().matmul(specific)
    # 计算相关性矩阵的 L1 范数（即矩阵元素的绝对值之和）
    norm = torch.norm(correlation_matrix, p=1)
    # 返回正交损失，鼓励共享表示和特定表示之间的正交性

    return norm
