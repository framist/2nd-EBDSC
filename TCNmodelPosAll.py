import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

""" Modern TCN 推荐 + 结构重参数化
统一的大小尺度特征 emb 输入

~~时间卷积使用'replicate'将矩阵的边缘复制并填充到矩阵的外围。~~
（原本是 zero 填充）
"""


class Embedding(nn.Module):
    def __init__(self, P=8, S=4, D=2048):
        super(Embedding, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, M, L]
        B = x.shape[0]
        x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
        x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
        )  # [B*M, 1, L] -> [B*M, 1, L+P-S]

        x_emb = self.conv(x_pad)  # [B*M, 1, L+P-S] -> [B*M, D, N]
        # [B*M, D, N] -> [B, M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)

        return x_emb  # x_emb: [B, M, D, N]


class EmbeddingPos(nn.Module):
    def __init__(self, pos_D, P=1, S=1, D=256):
        super(EmbeddingPos, self).__init__()
        self.P = P
        self.S = S
        self.conv = nn.Conv1d(
            in_channels=pos_D,
            out_channels=D,
            kernel_size=P,
            stride=S
        )

    def forward(self, x):
        # x: [B, M, pos_D, L]
        B = x.shape[0]
        # [B, M, pos_D, L] -> [B*M, pos_D, L]
        x = rearrange(x, 'b m r l -> (b m) r l')
        x_pad = F.pad(
            x,
            pad=(0, self.P-self.S),
            mode='replicate'
        )  # [B*M, pos_D, L] -> [B*M, pos_D, L+P-S]

        x_emb = self.conv(x_pad)  # [B*M, pos_D, L+P-S] -> [B*M, D, N]
        # [B*M, D, N] -> [B, M, D, N]
        x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)

        return x_emb  # x_emb: [B, M, D, N]


class ConvFFN(nn.Module):
    def __init__(self, M, D, r, one=True):  # one is True: ConvFFN1, one is False: ConvFFN2
        super(ConvFFN, self).__init__()
        groups_num = M if one else D
        self.pw_con1 = nn.Conv1d(
            in_channels=M*D,
            out_channels=r*M*D,
            kernel_size=1,
            groups=groups_num
        )
        self.pw_con2 = nn.Conv1d(
            in_channels=r*M*D,
            out_channels=M*D,
            kernel_size=1,
            groups=groups_num
        )

    def forward(self, x):
        # x: [B, M*D, N]
        x = self.pw_con2(F.gelu(self.pw_con1(x)))
        return x  # x: [B, M*D, N]

# backbone 的主结构


class ModernTCNBlock(nn.Module):
    def __init__(self, M, D, kernel_size, r):
        super(ModernTCNBlock, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.dw_conv = nn.Conv1d(
            in_channels=M*D,
            out_channels=M*D,
            kernel_size=kernel_size,
            groups=M*D,
            padding='same',
        )
        # 结构重参数化 small size = 5
        self.dw_conv_small = nn.Conv1d(
            in_channels=M*D,
            out_channels=M*D,
            kernel_size=5,
            groups=M*D,
            padding='same'
        )
        self.bn = nn.BatchNorm1d(M*D)
        self.bn_samll = nn.BatchNorm1d(M*D)
        self.conv_ffn1 = ConvFFN(M, D, r, one=True)
        self.conv_ffn2 = ConvFFN(M, D, r, one=False)

    def forward(self, x_emb):
        # x_emb: [B, M, D, N] 特征嵌入（也就是你的特征可以做嵌入操作，然后给网络来更好的处理）
        D = x_emb.shape[-2]
        # [B, M, D, N] -> [B, M*D, N]
        x = rearrange(x_emb, 'b m d n -> b (m d) n')
        
        # TODO 结构重参
        # [B, M*D, N] -> [B, M*D, N]
        x_l = self.bn(self.dw_conv(x))
        x_s = self.bn_samll(self.dw_conv_small(x))
        x = x_l + x_s

        # [B, M*D, N] -> [B, M*D, N]
        x = self.conv_ffn1(x)

        # [B, M*D, N] -> [B, M, D, N]
        x = rearrange(x, 'b (m d) n -> b m d n', d=D)
        # [B, M, D, N] -> [B, D, M, N]
        x = x.permute(0, 2, 1, 3)
        # [B, D, M, N] -> [B, D*M, N]
        x = rearrange(x, 'b d m n -> b (d m) n')

        # [B, D*M, N] -> [B, D*M, N]
        x = self.conv_ffn2(x)

        # [B, D*M, N] -> [B, D, M, N]
        x = rearrange(x, 'b (d m) n -> b d m n', d=D)
        # [B, D, M, N] -> [B, M, D, N]
        x = x.permute(0, 2, 1, 3)

        out = x + x_emb

        return out  # out: [B, M, D, N] 和 TCN 一样输入输出相同


'''
文件说明：
1. TCN 为 TCN 的预测任务，输入为 x: [B, M, L] （可以使用 embedding or 不使用直接看情况）

2. ModernTCN_DC 脉冲流标注任务：
    输入为 (64,5,1024) 5 个时序特征，长度 1024
    输出为 (64,1024,12),1024 长度序列对应 12 个类别

3. ModernTCN_mnist 为 TCN 的分类任务)

参数说明：
# B：batch size 😀
# M：多变量序列的变量数 😀 我的为 5  channel
# L：过去序列的长度 😀 我的为 1024  input
# T: 预测序列的长度 😀 我要改成分类任务 这里用不到 output

# embedding 参数：
# N: 分 Patch 后 Patch 的个数
# D：每个变量的通道数
# P：kernel size of embedding layer
# S：stride of embedding layer
'''

# TCN 最新工作 ModernTCN ,其实，可以写修改的版本，还方便测试


class ModernTCN_DC(nn.Module):  # T 在预测任务当中为预测的长度，可以更换为输出的种类 num_classes
    def __init__(self, M, L, num_classes, D=64, P=1, S=1, kernel_size=51, r=2, num_layers=24, pos_D=128):  # 如果能收敛就一点一点增加，在原来跑通的里面层数为
        # M, L, num_classes,
        super(ModernTCN_DC, self).__init__()
        # 深度分离卷积负责捕获时域关系
        self.num_layers = num_layers
        # N = L // S
        # 经验证，增加 emb 的话会增加模型震荡
        # self.embed_layer = EmbeddingPos(pos_D, P, S, D)
        # self.embed_layer = Embedding(P, S, D)

        self.backbone = nn.ModuleList(
            [ModernTCNBlock(M, D, kernel_size, r) for _ in range(num_layers)])

        # [B, M, D, N] -> [B, M, D, N]
        # rearrange 实现 [B, M, D, N] -> [B, M, D*N] 【64,5,D*N】 (D = 64)
        # D*N# N: 分 Patch 后 Patch 的个数# D：每个变量的通道数 D*N 😀 输出为你的输出 1024 原序列长度，然后接分类的任务和 loss 去做，具体可以参考 TCN 中分类任务当中任务头怎么写的（注，之前的工作可能没有特征 embedding 的映射流程，但任务设计和修改是好的）

        # w/o pool
        self.classificationhead = nn.Linear(D * M, num_classes)

        # # with pool
        # self.classificationhead = nn.Linear(D, num_classes)

    def forward(self, pos):
        # L = N = 1024 序列长 (P=1, S=1 时)
        # B = batch size
        # pos: [B, L=1024, M=5, pos_D=128]

        # [B, L=1024, M=5, pos_D=128] -> [B, M=5, D=128, L=1024]
        x_emb = rearrange(pos, 'b l m d -> b m d l')
        # x_emb = self.embed_layer(x_emb) # [B, L=1024, M=5, pos_D=128] -> [B, M=5, D=128, L=1024]

        for i in range(self.num_layers):  # 过几层 block
            x_emb = self.backbone[i](x_emb)  # [B, M, D, N] -> [B, M, D, N]

        # 在展平之前，[64, 5, 64, 1024] 要做序列标注任务 则 [64,5,1024,12] 将 5 个特征维度聚合得到 [64,1024,12]
        # 本质是 [B, M, D, N] -> [B, L, classes],其中 L 为 1024，classes 为 12，且 N = L // S
        # 可以考虑使用更复杂的池化方式、添加 dropout 等来增强模型的表达能力。

        # Flatten 将预测的长度拉开，把嵌入的维度拉开
        # [B, M, D, N] -> [B, M*D, N]
        cls1 = rearrange(x_emb, 'b m d n -> b (m d) n')

        # maxpool
        # cls1 = torch.max(x_emb, dim=1)[0]    # [B, M, D, N] -> [B, D, N]

        # 转换为 [64, 1024, 64]
        cls1 = cls1.permute(0, 2, 1)  # [64, 64, 1024] -> [64, 1024, 64]

        # 输出为 [64, 1024,12]
        # [64, 1024, 1, 64] -> [64, 1024, 1, 12]
        out1 = self.classificationhead(cls1)

        return out1


# 测试单元
if __name__ == '__main__':
    # (64,5,1024) 测试

    past_series = torch.rand(20, 1024, 5, 128).cuda()

    # model = ModernTCN_mnist(5, 1024, 10)  # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192

    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    model = ModernTCN_DC(5, 1024, 12, 128).cuda()

    pred_series = model(past_series)
    print(pred_series.shape)
