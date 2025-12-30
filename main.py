import csv
import random
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import dataset_with_info
from models import MvAEModel
from utils import Logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.metric import compute_metric

# 测试torch版本和CUDA是否可用
print(f'1.1 torch version:{torch.__version__}\n'
      f'1.2 cuda available:{torch.cuda.is_available()}')


# 设置随机数种子以保证结果的可复现性
def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# 定义超参数类，用于集中管理所有的超参数
class Args:
    def __init__(self):
        # 随机数种子
        self.seed = 42
        # 数据集文件夹地址
        self.folder_path = "data"  # 不要改
        # 训练和测试的批量大小
        self.train_batch_size = 10000
        self.test_batch_size = 10000
        # 训练的总轮数
        self.train_epoch = 50
        # 学习率
        self.lr = 0.001
        # 缺失处理的比例
        self.missing_ratio = 0.0
        # 冲突处理的比例
        self.ratio_conflict = 0.0
        # 噪声处理的比例
        self.ratio_noise = 0.0
        # 高斯噪声的标准差
        self.sigma = 0.5
        # 损失函数中alpha参数，对应于正则化项权重 TODO: loss = loss_rec + args.lambda_ma * (loss_mi + loss_ad) + args.lambda_con * loss_con  # 计算总损失
        self.lambda_ma = 0.01  # TODO 1. lambda_ma
        # 损失函数中beta参数，对应于对比损失项权重
        self.lambda_con = 0.01  # TODO 2. lambda_con
        # 是否启用对比学习
        self.do_contrast = True  # TODO 3.do contrast or not
        # 特征维度
        self.feature_dim = 64  # Fixed
        # 设备选择（CUDA或CPU）
        self.device = "cuda:0"
        # 开始聚类的轮数
        self.clustering_epoch = 20
        # 邻居数量，稍后根据实例数设置
        self.neighbors_num = None  # TODO 4.1 根据ins_num在代码中设置
        # 正样本数量
        self.pos_num = 21
        # 负样本数量，稍后根据邻居数量设置
        self.neg_num = None  # TODO 4.2 根据neighbors_num在代码中设置


# 定义正交损失函数，用于计算共享特征和特定特征之间的正交性损失
def orthogonal_loss(shared, specific):
    _shared = shared.detach()  # 分离出共享特征的梯度
    _shared = _shared - _shared.mean(dim=0)  # 减去均值
    correlation_matrix = _shared.t().matmul(specific)  # 计算相关矩阵
    norm = torch.norm(correlation_matrix, p=1)  # 计算1范数作为损失
    return norm


if __name__ == '__main__':
    args = Args()  # 初始化超参数
    seed_setting(args.seed)  # 设置随机数种子

    # 创建日志文件夹并初始化数据集信息文件
    if not os.path.exists('logs'):
        os.makedirs('logs')
    file_datasetInfo = 'logs/datasetInfo.csv'
    headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
    with open(file_datasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # 写入表头
    print(f"{file_datasetInfo} has been created")  # 输出文件创建信息

    # 获取数据集文件夹中的所有数据集文件名
    file_names = os.listdir(args.folder_path)
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):  # 检查文件扩展名
            dataset_name = dataset_name[:-4]  # 移除文件扩展名以获得数据集名称
        else:
            continue  # 如果不是.mat文件，跳过此文件

        print(
            f'-------------------------------------start training：Dataname[{dataset_name}]-------------------------------------')

        # 确保日志文件夹存在，若不存在则创建
        logs_dir = 'logs'
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            print("Logs directory created at:", logs_dir)
        else:
            print("Logs directory already exists at:", logs_dir)
        logger = Logger.get_logger(__file__, dataset_name)  # 初始化日志记录器

        # 加载数据集，并获取数据集的相关信息
        dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(dataset_name, file_datasetInfo)
        index = np.arange(len(dataset))  # 创建索引数组
        dataset.addMissing(index, args.missing_ratio)  # 添加缺失数据
        dataset.addConflict(index, args.ratio_conflict)  # 添加冲突数据
        dataset.addNoise(index, args.ratio_noise, args.sigma)  # 添加噪声数据

        # TODO 根据实例数量设置 4.1邻居数量 和 4.2 负样本数量
        args.neighbors_num = int(ins_num / 4)
        args.neg_num = int((args.neighbors_num - args.pos_num - 1) / 2)

        # 创建训练和测试的数据加载器
        train_loader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=False)
        test_loader = DataLoader(dataset, batch_size=args.test_batch_size, shuffle=False)

        # 初始化评价指标的数组
        ACC_array = np.zeros((6, 6))
        NMI_array = np.zeros((6, 6))
        Purity_array = np.zeros((6, 6))
        ARI_array = np.zeros((6, 6))

        # 初始化模型、损失函数和优化器
        model = MvAEModel(input_dims, view_num, args.feature_dim, h_dims=[500, 200]).to(args.device)
        mse_loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        nbr_idx = []
        neg_idx = []
        losses = []

        # 计算每个视图的邻居和负样本索引
        for v in range(view_num):
            X_np = np.array(dataset.features[0][v])  # 获取第v个视图的数据
            nbrs_v = np.zeros((ins_num, args.pos_num - 1))  # 初始化正样本邻居数组
            neg_v = np.zeros((ins_num, args.neg_num))  # 初始化负样本邻居数组
            # TODO GMAE中对比约束的不是不同视图，而是最终表征中的邻居与否。
            nbrs = NearestNeighbors(n_neighbors=args.neighbors_num, algorithm='auto').fit(X_np)  # 计算邻居
            dis, idx = nbrs.kneighbors(X_np)  # 获取距离和索引
            for i in range(ins_num):
                for j in range(args.pos_num - 1):
                    nbrs_v[i][j] += idx[i][j + 1]  # 正样本邻居索引
                for j in range(args.neg_num):
                    neg_v[i][j] += idx[i][args.neighbors_num - j - 1]  # 负样本邻居索引
            nbr_idx.append(torch.LongTensor(nbrs_v))  # 将邻居索引转换为张量并添加到列表
            neg_idx.append(torch.LongTensor(neg_v))  # 将负样本索引转换为张量并添加到列表

        nbr_idx = torch.cat(nbr_idx, dim=-1)  # 将所有视图的正样本邻居索引拼接在一起
        neg_idx = torch.cat(neg_idx, dim=-1)  # 将所有视图的负样本邻居索引拼接在一起

        acc_list = []
        nmi_list = []
        pur_list = []

        # 开始训练模型
        for epoch in range(args.train_epoch):
            save_loss = True  # 记录是否保存损失
            for x, y, idx, pu in train_loader:
                optimizer.zero_grad()  # 清除梯度
                model.train()  # 设置模型为训练模式
                for v in range(view_num):
                    x[v] = x[v].to(args.device)  # 将输入数据传输到指定设备
                clustering = epoch > args.clustering_epoch  # 判断是否开始聚类
                hidden_share, hidden_specific, hidden, recs = model(x)  # 模型前向传播，获取隐藏表示和重构结果
                loss_rec = 0  # 初始化重构损失
                loss_mi = 0  # 初始化正交损失
                loss_ad = 0  # 初始化判别损失
                labels_true = torch.ones(x[0].shape[0]).long().to(args.device)  # 正样本标签
                labels_false = torch.zeros(x[0].shape[0]).long().to(args.device)  # 负样本标签
                for v in range(view_num):
                    loss_rec += mse_loss_fn(recs[v], x[v])  # 计算重构损失
                    loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])  # 计算正交损失
                    loss_ad += model.discriminators_loss(hidden_specific, v)  # 计算判别损失
                loss_con = 0  # 初始化对比损失
                # TODO 3. do_contrast
                if args.do_contrast:
                    for i in range(len(idx)):
                        index = idx[i]
                        hidden_positive = hidden[nbr_idx[index]]  # 获取正样本隐藏表示
                        positive = torch.exp(
                            torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_positive.detach()))  # 计算正样本相似度
                        negative_idx = neg_idx[index]
                        hidden_negative = hidden[negative_idx]  # 获取负样本隐藏表示
                        negative = torch.exp(
                            torch.cosine_similarity(hidden[i].unsqueeze(0), hidden_negative.detach())).sum()  # 计算负样本相似度
                        loss_con -= torch.log((positive / negative)).sum()  # 计算对比损失
                        torch.cuda.empty_cache()  # 清除缓存
                    loss_con = loss_con / len(idx)  # 归一化对比损失
                # TODO 1. lambda_ma  2. lambda_con
                loss = loss_rec + args.lambda_ma * (loss_mi + loss_ad) + args.lambda_con * loss_con  # 计算总损失
                losses.append(loss.item())  # 记录损失值
                loss.backward()  # 反向传播
                optimizer.step()  # 更新模型参数

            with torch.no_grad():  # 在测试阶段不计算梯度
                for x, y, idx, pu in test_loader:
                    for v in range(view_num):
                        x[v] = x[v].to(args.device)  # 将输入数据传输到指定设备
                    model.eval()  # 设置模型为评估模式
                    hidden_share, hidden_specific, hidden, recs = model(x)  # 模型前向传播，获取隐藏表示和重构结果
                    kmeans = KMeans(n_clusters=nc, n_init=50)  # 使用KMeans进行聚类
                    datas = hidden.clone().cpu().numpy()  # 将隐藏表示转换为numpy数组
                    y_pred = kmeans.fit_predict(datas)  # 进行聚类预测
                    label = np.array(y)  # 转换标签为numpy数组
                    ACC, NMI, Purity, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)  # 计算各项评估指标
                    info = {"epoch": epoch, "acc": ACC, "Nmi": NMI, "ari": ARI, "Purity": Purity, "Fscore": F_score,
                            "Precision": Precision, "recall": Recall}  # 打包评估结果
                    logger.info(str(info))  # 记录评估信息
