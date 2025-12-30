import csv
import random
import torch
import sklearn
import scipy
import numpy as np
from torch.utils.data import Dataset
from scipy.io import loadmat


def loadData(data_name):
    data = scipy.io.loadmat(data_name)
    features = data['X']
    gnd = data['Y']
    gnd = gnd.flatten()
    return features, gnd


class AnyDataset(Dataset):

    def __init__(self, dataname, data_path):
        self.features, self.gnd = loadData(f'{data_path}')
        self.v = self.features.shape[1]
        # 数据归一化
        for i in range(0, self.v):
            minmax = sklearn.preprocessing.MinMaxScaler()
            self.features[0][i] = minmax.fit_transform(self.features[0][i])
        # 单位矩阵
        self.iden = torch.tensor(np.identity(self.features[0][0].shape[0])).float()
        self.dataname = dataname

        self.X = self.features[0]

    def __len__(self):
        return self.gnd.shape[0]

    def __getitem__(self, idx):
        """
        动态构建输入特征列表，根据self.v决定特征数量
        """
        feature_list = [torch.from_numpy(np.array(self.features[0][i][idx], dtype=np.float32)) for i in range(self.v)]
        return feature_list, torch.from_numpy(np.array(self.gnd[idx])), torch.from_numpy(np.array(idx)), torch.from_numpy(np.array(self.iden[idx]))

    def addMissing(self, index, ratio):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            elements = list(range(self.v))  # 生成一个包含0到self.v-1的列表
            random.seed()  # 确保每次运行时生成不同的随机数
            length = random.randint(1, self.v - 1)  # views数量为随机选取的该列表的子集长度
            views = random.sample(elements, length)  # 从该列表中随机选取length个不重复的元素
            print(f'add missing[{i}]: {views}')
            for v in views:
                self.X[v][i] = 0
        print(f'1. Add Missing completed[ratio: {ratio}]')
        pass

    def addConflict(self, index, ratio):
        Y = self.gnd
        Y = np.squeeze(Y)
        if np.min(Y) == 1:
            Y = Y - 1
        Y = Y.astype(dtype=np.int64)
        num_classes = len(np.unique(Y))

        records = dict()
        for c in range(num_classes):
            i = np.where(Y == c)[0][0]
            temp = dict()
            for v in range(self.v):
                temp[v] = self.X[v][i]
            records[c] = temp
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            v = np.random.randint(self.v)
            self.X[v][i] = records[(Y[i] + 1) % num_classes][v]
        print(f'2. Add Conflict completed: {ratio}]')
        pass

    def addNoise(self, index, ratio, sigma):
        selects = np.random.choice(index, size=int(ratio * len(index)), replace=False)
        for i in selects:
            elements = list(range(self.v))
            random.seed()
            length = random.randint(1, self.v)
            views = random.sample(elements, length)
            for v in views:
                self.X[v][i] = np.random.normal(self.X[v][i], sigma)
        print(f'3. Add Noise completed: {ratio}]')
        pass


# TODO 1.数据集加载 Dataloader
def dataset_with_info(dataname, FileDatasetInfo, meta_path):
    data_path = f'./{meta_path}/' + dataname + '.mat'
    features, gnd = loadData(data_path)
    views = max(features.shape[0], features.shape[1])
    input_num = features[0][0].shape[0]
    datasetforuse = AnyDataset(dataname, data_path)
    nc = len(np.unique(gnd))
    input_dims = []
    for v in range(views):
        dim = features[0][v].shape[1]
        input_dims.append(dim)
    print("Data: " + dataname + ", number of data: " + str(input_num) + ", views: " + str(views) + ", clusters: " +
          str(nc) + ", each view: ", input_dims)

    # TODO 3.保存数据集信息
    with open(FileDatasetInfo, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入数据
        data = [dataname, str(input_num), str(views), str(nc), input_dims]
        writer.writerow(data)

    return datasetforuse, input_num, views, nc, input_dims, gnd
