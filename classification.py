import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from loss import orthogonal_loss, contrastive_loss
from models import GMAE
from utils import Logger
from utils.dataloader import dataset_with_info
from utils.metric import compute_metric
from utils.plot import plot_acc, print_metrics_table


# =======================================================================
# 设置随机种子，保证实验可复现
# =======================================================================
def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 禁用 cudnn 优化
    torch.backends.cudnn.deterministic = True  # 保证每次执行相同的结果
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


# =======================================================================
# 评估模型的性能，包括分类头和K-means
# =======================================================================
def evaluate_model(model, data_loader, nc, view_num, device):
    with torch.no_grad():  # 不需要计算梯度
        for x, y, idx, pu in data_loader:
            # 将输入数据移到指定设备
            for v in range(view_num):
                x[v] = x[v].to(device)
            model.eval()  # 设置模型为评估模式

            # 获取模型输出
            hidden_share, hidden_specific, hidden, recs, classes = model(x)
            label = np.array(y)

            # 使用K-means进行聚类并计算指标
            y_pred_2 = KMeans(n_clusters=nc, n_init=50).fit_predict(hidden.cpu().numpy())
            ACC2, NMI2, Purity2, ARI2, F_score2, Precision2, Recall2 = compute_metric(label, y_pred_2)

            # 使用分类头预测标签并计算指标
            y_pred = torch.argmax(classes, dim=1).detach().cpu().numpy()
            ACC, NMI, Purity, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)

        return ACC, NMI, Purity, ARI, F_score, Precision, Recall, ACC2, NMI2, Purity2, ARI2, F_score2, Precision2, Recall2


# =======================================================================
# 主程序
# =======================================================================
if __name__ == '__main__':

    # ===================================================================
    # 使用argparse解析命令行超参数
    # ===================================================================
    parser = argparse.ArgumentParser(description='GMAE Model Training')
    parser.add_argument('--logs_path', default='1.logs_classification', type=str, help='Path to save logs')
    parser.add_argument('--imgs_path', default='2.imgs_classification', type=str, help='Path to save imgs')
    parser.add_argument('--folder_path', default='dataset', type=str, help='Dataset folder path')
    parser.add_argument('--do_plot', default=True, type=bool, help='Whether to plot the results')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for training')
    # TODO 1.超参数
    parser.add_argument('--train_epoch', default=500, type=int, help='Number of training epochs') # 500
    parser.add_argument('--eval_interval', default=100, type=int, help='Interval for evaluation')
    parser.add_argument('--seed', default=42, type=int, help='Random seed')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dimensions')
    parser.add_argument('--lambda_ma', default=0.01, type=float, help='Lambda for mutual alignment loss')
    parser.add_argument('--lambda_con', default=0.01, type=float, help='Lambda for contrastive loss')
    parser.add_argument('--pos_num', default=21, type=int, help='Positive sample number')
    parser.add_argument('--do_contrast', default=True, type=bool, help='Whether to use contrastive loss')
    # TODO 2.数据处理
    parser.add_argument('--ratio_noise', default=0.0, type=float, help='Noise ratio')
    parser.add_argument('--ratio_conflict', default=0.0, type=float, help='Conflict ratio')
    parser.add_argument('--missing_ratio', default=0.0, type=float, help='Missing ratio')

    args = parser.parse_args()  # 解析参数
    seed_setting(args.seed)  # 设置随机种子

    # ===================================================================
    # 创建日志目录
    # ===================================================================
    if not os.path.exists(args.logs_path):
        os.makedirs(args.logs_path)
        print("Logs directory created at:", args.logs_path)
    else:
        print("Logs directory already exists at:", args.logs_path)

    # ===================================================================
    # 创建并写入数据集信息的CSV文件
    # ===================================================================
    file_datasetInfo = f'{args.logs_path}/datasetInfo.csv'
    headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
    with open(file_datasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"{file_datasetInfo} has been created")

    # ===================================================================
    # 遍历数据集文件夹并加载数据
    # ===================================================================
    file_names = os.listdir(args.folder_path)
    data_iter = 1 # 数据集位次
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):
            dataset_name = dataset_name[:-4]  # 去掉文件名后缀，得到数据集名称
            print(f'----------------------------------{dataset_name}[{data_iter}]----------------------------------')

            # 初始化日志记录器
            logger = Logger.get_logger(__file__, dataset_name, args.logs_path)

            # 获取数据集信息
            dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(
                dataset_name, file_datasetInfo, args.folder_path)

            # ===================================================================
            # 对数据集进行数据增强处理
            # ===================================================================
            index = np.arange(len(dataset))
            dataset.addMissing(index, args.missing_ratio)  # 添加缺失数据
            dataset.addConflict(index, args.ratio_conflict)  # 添加冲突数据
            dataset.addNoise(index, args.ratio_noise, sigma=0.5)  # 添加噪声

            # ===================================================================
            # 随机划分训练集和测试集
            # ===================================================================
            split_seed = int.from_bytes(os.urandom(8), "little")
            rng_split = np.random.default_rng(split_seed)
            index_dataset = np.arange(ins_num)
            rng_split.shuffle(index_dataset)
            split = int(0.8 * ins_num)
            train_index, test_index = index_dataset[:split], index_dataset[split:]

            # 设置DataLoader，提供训练数据和测试数据
            g = torch.Generator()
            g.manual_seed(args.seed)  # 设置训练过程的随机种子
            train_loader = DataLoader(Subset(dataset, train_index), batch_size=split, shuffle=True, generator=g)
            test_loader = DataLoader(Subset(dataset, test_index), batch_size=ins_num - split, shuffle=False)

            # ===================================================================
            # 初始化邻居信息
            # ===================================================================
            train_ins_num = len(train_index)
            neighbors_num = int(train_ins_num / 4)
            pos_num = args.pos_num
            neg_num = int((neighbors_num - pos_num - 1) / 2)
            nbr_idx, neg_idx = [], []

            # 获取邻居和负样本索引
            for v in range(view_num):
                X_np = np.array([dataset[i][0][v].numpy() if isinstance(dataset[i][0][v], torch.Tensor)
                                 else dataset[i][0][v] for i in train_index], dtype=object)
                if all(x.shape == X_np[0].shape for x in X_np):
                    X_np = np.vstack(X_np)
                nbrs_v, neg_v = np.zeros((train_ins_num, pos_num - 1)), np.zeros((train_ins_num, neg_num))
                nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
                dis, idx = nbrs.kneighbors(X_np)

                for i in range(train_ins_num):
                    nbrs_v[i][:] = idx[i][1:pos_num]
                    neg_v[i][:] = idx[i][-neg_num:]

                nbr_idx.append(torch.LongTensor(nbrs_v))
                neg_idx.append(torch.LongTensor(neg_v))

            # 拼接邻居和负样本索引
            nbr_idx = torch.cat(nbr_idx, dim=-1)
            neg_idx = torch.cat(neg_idx, dim=-1)

            # ===================================================================
            # 选择训练设备（GPU或CPU）
            # ===================================================================
            device = args.device
            h_dims = [500, 200]
            model = GMAE(input_dims, view_num, args.feature_dim, h_dims, nc).to(device)
            mse_loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            # ===================================================================
            # 记录性能指标
            # ===================================================================
            acc_list, nmi_list, pur_list, ari_list = [], [], [], []

            # ===================================================================
            # 训练过程
            # ===================================================================
            for epoch in tqdm(range(args.train_epoch)):
                epoch_loss = 0.0
                criterion = nn.CrossEntropyLoss()
                for x, y, train_idx, pu in train_loader:
                    optimizer.zero_grad()
                    model.train()

                    # 将数据移到设备
                    for v in range(view_num):
                        x[v] = x[v].to(device)

                    hidden_share, hidden_specific, hidden, recs, classes = model(x)
                    loss_rec, loss_mi, loss_ad, loss_class = 0, 0, 0, 0

                    if y.min() == 1:
                        y = (y - 1).long().to(device)
                    elif y.min() == 0:
                        y = y.long().to(device)

                    for v in range(view_num):
                        loss_rec += mse_loss_fn(recs[v], x[v])
                        loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])
                        loss_ad += model.discriminators_loss(hidden_specific, v)
                        loss_class += criterion(classes, y)

                    # 对比损失
                    loss_con = contrastive_loss(args, hidden, nbr_idx, neg_idx, train_idx)
                    total_loss = loss_rec + args.lambda_ma * (
                                loss_mi + loss_ad) + args.lambda_con * loss_con + loss_class
                    total_loss.backward()
                    optimizer.step()
                    epoch_loss += total_loss.item()
                print(f'\nTotal loss [Classification]: {epoch_loss}')

                # 每隔eval_interval轮进行评估
                if (epoch + 1) % args.eval_interval == 0:
                    # 训练集和测试集评估
                    acc_tr, nmi_tr, pur_tr, ari_tr, _, _, _, acc_tr2, nmi_tr2, pur_tr2, ari_tr2, _, _, _ = \
                        evaluate_model(model, train_loader, nc, view_num, device)
                    acc_te, nmi_te, pur_te, ari_te, _, _, _, acc_te2, nmi_te2, pur_te2, ari_te2, _, _, _ = \
                        evaluate_model(model, test_loader, nc, view_num, device)

                    # 打印和记录评估结果
                    print_metrics_table(
                        epoch + 1,
                        train_cls=(acc_tr, nmi_tr, pur_tr, pur_tr),
                        train_km=(acc_tr2, nmi_tr2, pur_tr2, pur_tr2),
                        test_cls=(acc_te, nmi_te, pur_te, pur_te),
                        test_km=(acc_te2, nmi_te2, pur_te2, pur_te2)
                    )

                    # 记录指标
                    acc_list.append(acc_te)
                    nmi_list.append(nmi_te)
                    pur_list.append(pur_te)
                    ari_list.append(ari_te)

                    info = {
                        "epoch": epoch + 1,
                        "acc": acc_te,
                        "nmi": nmi_te,
                        "ari": ari_te,
                        "pur": pur_te
                    }
                    logger.info(str(info))

            # ===================================================================
            # 绘图
            # ===================================================================
            if args.do_plot:
                plot_acc(acc_list, dataset_name, 'acc', args.imgs_path)
                plot_acc(nmi_list, dataset_name, 'nmi', args.imgs_path)
                plot_acc(pur_list, dataset_name, 'pur', args.imgs_path)
                plot_acc(ari_list, dataset_name, 'ari', args.imgs_path)

        else:
            print(f'Non-MAT file. Please convert the dataset to multi-view one-dimensional MAT format.')
        data_iter += 1