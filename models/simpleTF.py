import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias))
    result.add_module('bn', get_bn(out_channels))
    return result


def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, groups,
                 small_kernel,
                 small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish, and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=1, groups=groups, bias=True)
        else:
            self.lkb_origin = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=1, groups=groups, bias=False)
            if small_kernel is not None:
                assert small_kernel <= kernel_size, 'The kernel size for re-param cannot be larger than the large kernel!'
                self.small_conv = conv_bn(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=small_kernel,
                                          stride=stride, padding=small_kernel // 2, groups=groups, dilation=1, bias=False)

    def forward(self, inputs):

        if hasattr(self, 'lkb_reparam'):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, 'small_conv'):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left).cuda()
            pad_right = torch.zeros(D_out, D_in, pad_length_right).cuda()
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values

        x = torch.cat([pad_left, x], dim=-1)
        x = torch.cat([x, pad_right], dim=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, 'small_conv'):
            small_k, small_b = fuse_bn(
                self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(small_k, (self.kernel_size - self.small_kernel) // 2,
                                          (self.kernel_size - self.small_kernel) // 2, 0)
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(in_channels=self.lkb_origin.conv.in_channels,
                                     out_channels=self.lkb_origin.conv.out_channels,
                                     kernel_size=self.lkb_origin.conv.kernel_size, stride=self.lkb_origin.conv.stride,
                                     padding=self.lkb_origin.conv.padding, dilation=self.lkb_origin.conv.dilation,
                                     groups=self.lkb_origin.conv.groups, bias=True)
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__('lkb_origin')
        if hasattr(self, 'small_conv'):
            self.__delattr__('small_conv')


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(in_channels=nvars * dmodel, out_channels=nvars * dmodel,
                                         kernel_size=large_size, stride=1, groups=nvars * dmodel,
                                         small_kernel=small_size, small_kernel_merged=small_kernel_merged, nvars=nvars)
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1,
                                 padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff//dmodel

    def forward(self, x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M*D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B*M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, nvars,
                 small_kernel_merged=False, drop=0.):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for _ in range(num_blocks):
            blk = Block(large_size=large_size, small_size=small_size, dmodel=dmodel,
                        dff=d_ffn, nvars=nvars, small_kernel_merged=small_kernel_merged, drop=drop)
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCNnew(nn.Module):  # T 在预测任务当中为预测的长度，可以更换为输出的种类 num_classes
    def __init__(self, M, num_classes, D=128, large_sizes=51, ffn_ratio=2, num_layers=24, 
                 small_size=5, small_kernel_merged=False, backbone_dropout=0., head_dropout=0., stem=False):  # 如果能收敛就一点一点增加，在原来跑通的里面层数为
        # M, L, num_classes,
        super(ModernTCNnew, self).__init__()
        self.num_layers = num_layers
        
        # # RevIN
        # self.revin = revin
        # if self.revin: self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer
        if stem:
            self.stem = nn.Sequential(
                nn.Conv1d(1, D, kernel_size=1, stride=1),
                nn.BatchNorm1d(D)
            )


        # backbone
        self.stages = Stage(ffn_ratio, num_layers, large_size=large_sizes, small_size=small_size, dmodel=D,
                            nvars=M, small_kernel_merged=small_kernel_merged, drop=backbone_dropout)

        # w/o pool
        # self.classificationhead = nn.Linear(D * M, num_classes)
        self.classificationhead = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(D * M, num_classes)
        )

        # # with pool
        # self.classificationhead = nn.Linear(D, num_classes)

    def forward(self, x: torch.Tensor):
        # L = N = 1024 序列长 (P=1, S=1 时)
        # B = batch size
        
        if hasattr(self, 'stem'):
            # x: [B, L=1024, M=5] -> [B, M=5, L]
            B = x.shape[0]
            x = rearrange(x, 'b l m -> b m l')
            x = x.unsqueeze(2)  # [B, M, L] -> [B, M, 1, L]
            x = rearrange(x, 'b m r l -> (b m) r l')  # [B, M, 1, L] -> [B*M, 1, L]
            x_emb = self.stem(x)
            x_emb = rearrange(x_emb, '(b m) d n -> b m d n', b=B)  # [B*M, D, N] -> [B, M, D, N]
        else:            
            # x: [B, L=1024, M=5, pos_D=128] -> [B, M=5, D=128, L=1024]
            x_emb = rearrange(x, 'b l m d -> b m d l')
        
        x_emb = self.stages(x_emb)

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

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, 'merge_kernel'):
                m.merge_kernel()


if __name__ == '__main__':
    from time import time

    past_series = torch.rand(10, 1024, 5, 128).cuda()
    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    model = ModernTCNnew(5, 12).cuda()

    start = time()
    pred_series = model(past_series)
    end = time()
    print(pred_series.shape, f"time {end - start}")

    model.structural_reparam()

    start = time()
    pred_series = model(past_series)
    end = time()

    print(pred_series.shape, f"time {end - start}")


    past_series2 = torch.rand(10, 1024, 5).cuda()
    # 对应的参数含义为 M, L, T, 4 个序列特征，96 原输入长度 96，预测输出长度为 192
    model = ModernTCNnew(5, 12, stem=True).cuda()

    start = time()
    pred_series = model(past_series2)
    end = time()
    print(pred_series.shape, f"time {end - start}")

    model.structural_reparam()

    start = time()
    pred_series = model(past_series2)
    end = time()

    print(pred_series.shape, f"time {end - start}")
