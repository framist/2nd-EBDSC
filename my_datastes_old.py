import torch
import torch.utils.data as Data
import numpy as np


class MyDataSet(Data.Dataset):
    """自定义数据集函数"""

    def __init__(self, inputs, targets, hard=None, pos_d=128, if_emb=True):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
        self.targets = targets
        
        if not if_emb:
            return
        
        self.d_model = pos_d
        self.d_step = 8
        # mod_max = 65536 = 2**N 因为满足 2 ** (N*d_step/d_model) == 2
        mod_max = 65536
        # div_term
        # self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)).to(device)
        # self.div_term = (1. / (65536 ** (torch.arange(0, d_model, 2) / d_model))).to(device)
        self.div_term = (
            1. / (mod_max ** (torch.arange(0, self.d_model, self.d_step) / self.d_model)))
        # mod_d = lambda d: mod_max ** (np.floor(d/d_step)*d_step / d_model)
        # 根据 mod 查维度，上界 <
        self.d_mod = lambda m: np.floor(
            self.d_step * np.log2(m)).astype(np.int64) + 1
        self.hard = hard

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        
        if not hasattr(self, 'd_model'):
            inputs = self.inputs[idx] / 65536 # TODO
            return inputs, self.targets[idx]

        # * POS [1024, 5] -> [1024, 5, 128]
        positions: torch.FloatTensor = self.inputs[idx]  # [1024, 5]
        win_size, input_channels = positions.size()
        pe = torch.zeros(win_size, input_channels, self.d_model)

        positions = positions.unsqueeze(-1)  # [1024, 5, 1]

        # pe[:, :, 0::2] = torch.sin(positions * self.div_term * torch.pi)
        # pe[:, :, 1::2] = torch.cos(positions * self.div_term * torch.pi)

        for i in range(self.d_step):
            pe[:, :, i::self.d_step] = (
                positions * self.div_term + 1 / self.d_step * i) % 1 * 2 - 1
            # # linearV
            # pe[:, :, i::self.d_step] = torch.absolute((positions * self.div_term + 1 / self.d_step * i) % 1 - 0.5) * 2 - 1

        if self.hard:
            # 初步测试 rand 和 mean 的效果差不多 acc~87 loss~.35
            def f_mask(x): return torch.mean(x, axis=0)
            # f_mask = lambda x: torch.rand_like(x) * 2 - 1
            # f_mask = lambda x: torch.zeros_like(x)
            if np.random.rand() < 1 * self.hard:
                # 1 RF mimax in 5 ~ 28
                mask_d_min = np.random.randint(self.d_mod(5), self.d_mod(60))
                pe[:, 1, mask_d_min:] = f_mask(pe[:, 1, mask_d_min:])

            if np.random.rand() < 1 * self.hard:
                # 2 PW * 10 mimax in 6 ~ 50
                mask_d_min = np.random.randint(self.d_mod(6), self.d_mod(100))
                pe[:, 2, mask_d_min:] = f_mask(pe[:, 2, mask_d_min:])

            if np.random.rand() < 0.1 * self.hard:
                # 3 RF ?
                pe[:, 3, :] = f_mask(pe[:, 3, :])

            if np.random.rand() < 0.5 * self.hard:
                # 4 DOA mimax in 6 ~ 7
                mask_d_min = np.random.randint(self.d_mod(6), self.d_mod(14))
                pe[:, 4, mask_d_min:] = f_mask(pe[:, 4, mask_d_min:])

        return pe, self.targets[idx]


class MyDataSet_woEmb(MyDataSet):
    """自定义数据集函数"""

    def __init__(self, inputs, targets, hard=None, pos_d=128):
        super(MyDataSet_woEmb, self).__init__(inputs, targets, hard, pos_d, False)