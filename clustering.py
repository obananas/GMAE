import argparse
import csv
import random
import torch
import os
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import orthogonal_loss, contrastive_loss
from utils.dataloader import dataset_with_info
from models import GMAE_MVC
from utils import Logger
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from utils.metric import compute_metric
from utils.plot import plot_acc


def seed_setting(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def prepare_neighbors(dataset, ins_num, view_num):
    neighbors_num = int(ins_num / 4)
    pos_num = 21
    neg_num = int((neighbors_num - pos_num - 1) / 2)
    nbr_idx, neg_idx = [], []

    for v in range(view_num):
        X_np = np.array(dataset.features[0][v])
        nbrs_v, neg_v = np.zeros((ins_num, pos_num - 1)), np.zeros((ins_num, neg_num))
        nbrs = NearestNeighbors(n_neighbors=neighbors_num, algorithm='auto').fit(X_np)
        dis, idx = nbrs.kneighbors(X_np)
        for i in range(ins_num):
            nbrs_v[i][:] = idx[i][1:pos_num]
            neg_v[i][:] = idx[i][-neg_num:]
        nbr_idx.append(torch.LongTensor(nbrs_v))
        neg_idx.append(torch.LongTensor(neg_v))

    nbr_idx = torch.cat(nbr_idx, dim=-1)
    neg_idx = torch.cat(neg_idx, dim=-1)
    return nbr_idx, neg_idx


def train_one_epoch(args, model, mse_loss_fn, optimizer, dataset, ins_num, view_num, nbr_idx, neg_idx, epoch, device):
    train_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)

    for x, y, idx, pu in train_loader:
        optimizer.zero_grad()
        model.train()

        for v in range(view_num):
            x[v] = x[v].to(device)

        hidden_share, hidden_specific, hidden, recs = model(x)
        loss_rec, loss_mi, loss_ad = 0, 0, 0

        for v in range(view_num):
            loss_rec += mse_loss_fn(recs[v], x[v])
            loss_mi += orthogonal_loss(hidden_share, hidden_specific[v])
            loss_ad += model.discriminators_loss(hidden_specific, v)

        loss_con = contrastive_loss(args, hidden, nbr_idx, neg_idx, idx)
        total_loss = loss_rec + args.lambda_ma * (loss_mi + loss_ad) + args.lambda_con * loss_con
        total_loss.backward()
        optimizer.step()


def evaluate_model(model, dataset, nc, ins_num, view_num, device):
    test_loader = DataLoader(dataset, batch_size=ins_num, shuffle=False)

    with torch.no_grad():
        for x, y, idx, pu in test_loader:
            for v in range(view_num):
                x[v] = x[v].to(device)
            model.eval()
            hidden_share, hidden_specific, hidden, recs = model(x)
            y_pred = KMeans(n_clusters=nc, n_init=50).fit_predict(hidden.cpu().numpy())
            label = np.array(y)
            ACC, NMI, Purity, ARI, F_score, Precision, Recall = compute_metric(label, y_pred)
            return ACC, NMI, Purity, ARI, F_score, Precision, Recall


if __name__ == '__main__':
    # ===================================================================
    # 使用argparse解析命令行超参数
    # ===================================================================
    parser = argparse.ArgumentParser(description='GMAE Model Training')
    parser.add_argument('--logs_path', default='1.logs_clustering', type=str, help='Path to save logs')
    parser.add_argument('--imgs_path', default='2.imgs_clustering', type=str, help='Path to save imgs')
    parser.add_argument('--folder_path', default='dataset', type=str, help='Dataset folder path')
    parser.add_argument('--do_plot', default=True, type=bool, help='Whether to plot the results')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use for training')
    # TODO 1.超参数
    parser.add_argument('--train_epoch', default=500, type=int, help='Number of training epochs') # 500
    parser.add_argument('--eval_interval', default=10, type=int, help='Interval for evaluation')
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

    file_datasetInfo = f'{args.logs_path}/datasetInfo.csv'

    headers = ['Dataname', 'number of data', 'views', 'clusters', 'each view']
    with open(file_datasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
    print(f"{file_datasetInfo} has been created")

    file_names = os.listdir(args.folder_path)
    data_iter = 1  # 数据集位次
    for dataset_name in file_names:
        if dataset_name.endswith(".mat"):
            dataset_name = dataset_name[:-4]
            print(f'----------------------------------{dataset_name}[{data_iter}]----------------------------------')

            logger = Logger.get_logger(__file__, dataset_name, args.logs_path)
            dataset, ins_num, view_num, nc, input_dims, _ = dataset_with_info(
                dataset_name, file_datasetInfo, args.folder_path)

            model = GMAE_MVC(input_dims, view_num, args.feature_dim, h_dims=[500, 200]).to(args.device)
            mse_loss_fn = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            nbr_idx, neg_idx = prepare_neighbors(dataset, ins_num, view_num)
            acc_list, nmi_list, pur_list, ari_list = [], [], [], []

            for epoch in tqdm(range(args.train_epoch)):
                train_one_epoch(args, model, mse_loss_fn, optimizer, dataset, ins_num, view_num, nbr_idx, neg_idx,
                                epoch, args.device)

                # 每隔 `eval_interval` 轮测试一次
                if (epoch + 1) % args.eval_interval == 0:
                    acc, nmi, pur, ari, _, _, _ = evaluate_model(model, dataset, nc, ins_num, view_num, args.device)
                    acc_list.append(acc)
                    nmi_list.append(nmi)
                    pur_list.append(pur)
                    ari_list.append(ari)

                    info = {"epoch": epoch + 1, "acc": acc, "nmi": nmi, "ari": ari, "pur": pur}
                    logger.info(str(info))

            plot_acc(acc_list, dataset_name, 'acc', args.imgs_path)
            plot_acc(nmi_list, dataset_name, 'nmi', args.imgs_path)
            plot_acc(pur_list, dataset_name, 'pur', args.imgs_path)
            plot_acc(ari_list, dataset_name, 'ari', args.imgs_path)
        else:
            print(f'Non-MAT file. Please convert the dataset to multi-view one-dimensional MAT format.')
        data_iter += 1
