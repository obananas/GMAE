import torch
import torch.nn as nn


# 编码器（Encoder）类
class Encoder(nn.Module):
    def __init__(self, dims):
        super(Encoder, self).__init__()
        self.dims = dims # (4层)编码器各layer的维度列表 [(view) input dims, 500, 200, 128]
        models = []
        # 1. 构建编码器的各层
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1]))
            # 2. 对1,2,4层应用 ReLU 激活函数
            if i != len(self.dims) - 2:
                models.append(nn.ReLU())
            # 3. 第3层使用 Dropout 防止过拟合
            else:
                models.append(nn.Dropout(p=0.5))
                # 归一化潜在表示和稳定训练：避免某些视图在损失函数计算时占主导地位
                models.append(nn.Softmax(dim=1))
        self.models = nn.Sequential(*models)

    def forward(self, X):
        # 前向传播：通过顺序模型传递数据
        return self.models(X)


# 解码器（Decoder）类
class Decoder(nn.Module):
    def __init__(self, dims): # (4层)解码器各layer的维度列表 [256, 200, 500, (view) input dims]
        super(Decoder, self).__init__()
        self.dims = dims
        models = []
        # 1. 构建解码器的各层
        for i in range(len(self.dims) - 1):
            models.append(nn.Linear(self.dims[i], self.dims[i + 1])) # 添加全连接层
            # 2. 对第3层使用 Dropout 防止过拟合
            if i == len(self.dims) - 2:
                models.append(nn.Dropout(p=0.5))
                models.append(nn.Sigmoid()) # 输出(0-1)概率
            # 3. 对1,2,4层应用 ReLU 激活函数
            else:
                models.append(nn.ReLU())
        self.models = nn.Sequential(*models)

    def forward(self, X):
        # 前向传播：通过顺序模型传递数据
        return self.models(X)


# 判别器（Discriminator）类
class Discriminator(nn.Module):
    def __init__(self, input_dim, feature_dim=64):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim # 输入维度
        self.feature_dim = feature_dim # 特征维度
        # 1. 定义判别器的网络结构
        self.discriminator = nn.Sequential(
            nn.Linear(self.input_dim, self.feature_dim), # 输入层到隐藏层
            nn.LeakyReLU(), # 激活函数
            nn.Linear(self.feature_dim, 1), # 输出层
            nn.Sigmoid() # 输出(0-1)概率
        )

    def forward(self, x):
        # 前向传播：通过顺序模型传递数据
        return self.discriminator(x)


# 计算判别器的损失函数
def discriminator_loss(real_out, fake_out, lambda_dis=1):
    # 1. 计算真实样本损失，目标是 1
    real_loss = nn.BCEWithLogitsLoss()(real_out, torch.ones_like(real_out))
    # 2. 计算假样本损失，目标是 0
    fake_loss = nn.BCEWithLogitsLoss()(fake_out, torch.zeros_like(fake_out))
    # 3. 返回总损失，乘以超参数 lambda_dis
    return lambda_dis * (real_loss + fake_loss)


# TODO 1. GMAE 多视图分类模型
class GMAE(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims, num_classes):
        # 初始化模型的基本结构
        super().__init__()
        self.input_dims = input_dims  # 各个视图的输入维度列表
        self.view_num = view_num  # 视图的数量
        self.out_dims = out_dims  # 每个视图输出的维度
        self.h_dims = h_dims  # 隐层维度
        self.discriminators = nn.ModuleList()  # 用于判别器的列表

        # 为每个视图创建一个判别器
        for v in range(view_num):
            self.discriminators.append((Discriminator(out_dims)))
        # 反向排列的隐藏层维度
        h_dims_reverse = list(reversed(h_dims))

        self.encoders_specific = nn.ModuleList() # 视图编码器
        self.decoders_specific = nn.ModuleList() # 视图解码器

        # 为每个视图创建独立的编码器和解码器
        for v in range(self.view_num):
            # 编码器：输入为当前视图的维度，输出为 `out_dims`
            self.encoders_specific.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            # 解码器：输入为 `out_dims*2` 和反向排列的隐藏层维度，输出为原始视图的维度
            self.decoders_specific.append(Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]))
        # 计算输入维度的总和，用于共享编码器
        d_sum = 0
        for d in input_dims:
            d_sum += d
        # 创建共享编码器，输入为所有视图的拼接维度，输出为 `out_dims`
        self.encoder_share = Encoder([d_sum] + h_dims + [out_dims])

        # 创建分类头：输出类别数
        hidden_dim = out_dims * (view_num + 1) # 隐藏层维度，包含共享表示和每个视图的特定表示
        mid = min(256, hidden_dim) # 中间层维度，避免过大
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, mid),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(mid, num_classes),  # 最终分类输出
        )

        # 创建 LayerNorm 层
        self.block_norm = nn.LayerNorm(out_dims)

    # 计算判别器的损失函数
    def discriminators_loss(self, hidden_specific, i, lambda_dis=1):
        discriminate_loss = 0.
        # 1. 遍历每个视图，计算其损失
        for j in range(self.view_num):
            if j != i:
                real_out = self.discriminators[i](hidden_specific[i])  # 当前视图判别器预测真实输出
                fake_out = self.discriminators[i](hidden_specific[j])  # 其他视图的判别器预测假输出
                # 计算判别损失并累加
                discriminate_loss += discriminator_loss(real_out, fake_out, lambda_dis)
        return discriminate_loss

    def forward(self, x_list):
        # 1. 拼接所有视图特征
        x_total = torch.cat(x_list, dim=-1)
        # 2. 计算共享潜表示
        hidden_share = self.encoder_share(x_total)
        recs = []  # 存储每个视图的重构结果
        hidden_specific = []  # 存储每个视图的特定潜表示

        # 3. 逐视图进行编码和解码
        for v in range(self.view_num):
            x = x_list[v]
            hidden_specific_v = self.encoders_specific[v](x)
            hidden_specific.append(hidden_specific_v)
            # 将共享潜在表示和当前视图的特定潜在表示拼接
            hidden_v = torch.cat((hidden_share, hidden_specific_v), dim=-1)
            rec = self.decoders_specific[v](hidden_v)
            recs.append(rec)

        # 4. 对共享表示和每个视图的特定表示进行整合
        hidden_list = [self.block_norm(hidden_share)] + [self.block_norm(h) for h in hidden_specific]
        hidden = torch.cat(hidden_list, dim=-1)  # 拼接所有表示

        # 5. 使用分类器进行分类
        class_output = self.classifier(hidden)

        return hidden_share, hidden_specific, hidden, recs, class_output

# TODO 2. GMAE_MVC 多视图聚类模型类
class GMAE_MVC(nn.Module):
    def __init__(self, input_dims, view_num, out_dims, h_dims):
        super().__init__()
        self.input_dims = input_dims
        self.view_num = view_num
        self.out_dims = out_dims
        self.h_dims = h_dims
        self.discriminators = nn.ModuleList()
        for v in range(view_num):
            self.discriminators.append((Discriminator(out_dims)))
        h_dims_reverse = list(reversed(h_dims))
        self.encoders_specific = nn.ModuleList()
        self.decoders_specific = nn.ModuleList()
        for v in range(self.view_num):
            self.encoders_specific.append(Encoder([input_dims[v]] + h_dims + [out_dims]))
            self.decoders_specific.append(Decoder([out_dims * 2] + h_dims_reverse + [input_dims[v]]))
        d_sum = 0
        for d in input_dims:
            d_sum += d
        self.encoder_share = Encoder([d_sum] + h_dims + [out_dims])

    def discriminators_loss(self, hidden_specific, i, LAMB_DIS=1):
        discriminate_loss = 0.
        for j in range(self.view_num):
            if j != i:
                real_out = self.discriminators[i](hidden_specific[i])
                fake_out = self.discriminators[i](hidden_specific[j])
                discriminate_loss += discriminator_loss(real_out, fake_out, LAMB_DIS)
        return discriminate_loss

    def forward(self, x_list):
        x_total = torch.cat(x_list, dim=-1)
        hidden_share = self.encoder_share(x_total)
        recs = []
        hidden_specific = []
        for v in range(self.view_num):
            x = x_list[v]
            hidden_specific_v = self.encoders_specific[v](x)
            hidden_specific.append(hidden_specific_v)
            hidden_v = torch.cat((hidden_share, hidden_specific_v), dim=-1)
            rec = self.decoders_specific[v](hidden_v)
            recs.append(rec)
        hidden_list = [hidden_share] + hidden_specific
        hidden = torch.cat(hidden_list, dim=-1)

        return hidden_share, hidden_specific, hidden, recs

