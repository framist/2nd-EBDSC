# %%
# Modern TCN 推荐参数
import numpy as np
from tqdm import tqdm
from typing import List
import torch
from torch import nn, Tensor
import torch.utils.data as Data

# %matplotlib widget
from typing import List
import datetime
now = datetime.datetime.now().strftime('%b%d_%H-%M')


# plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 中文字体设置
# plt.rcParams['axes.unicode_minus'] = False  # 负号显示设置

"""
Hard ~.8
TCN_amp_128D24L2R_POS_woRFnoise_mloss_0.333_cp-221.pth -> TCN_linear_mask_128D24L2R_POS_ALL_mloss_0.325_cp-209.pth
Pre: 29.752, Rec: 27.866, Acc: 68.288 -> Pre: 34.004, Rec: 26.359, Acc: 70.955
Pre: 62.469, Rec: 58.693, Acc: 81.622 -> Pre: 57.910, Rec: 54.157, Acc: 82.411
Pre: 89.584, Rec: 89.008, Acc: 88.563 -> Pre: 88.007, Rec: 87.897, Acc: 88.242
提交测试：0.6912 -> 0.7119

Easy: (.2/.2/.1/.2)
TCN_amp_128D24L2R_POS_woRFnoise_mloss_0.333_cp-221.pth -> TCN_128D24L2R_PosAll_Easy_mloss_0.286_cp-297.pth
Pre: 29.752, Rec: 27.866, Acc: 68.288 -> Pre: 48.079, Rec: 34.512, Acc: 67.521
Pre: 62.469, Rec: 58.693, Acc: 81.622 -> Pre: 62.987, Rec: 58.754, Acc: 81.976
Pre: 89.584, Rec: 89.008, Acc: 88.563 -> Pre: 89.700, Rec: 89.632, Acc: 90.007

目前最优 baseline：先 StepLR，再 CosineAnnealingLR
"""

# %% [markdown]
# 命令行参数示例：`python tcn_pos_all.py --cuda 4 --batch_size 50 --num_layers 24 --ratio 2`
import argparse
parser = argparse.ArgumentParser(description='Code for my modernTCN -- by framist',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cuda', type=int, default=0, help='所使用的 cuda 设备，暂不支持多设备并行')
parser.add_argument('--num_layers', type=int, default=24, help='layers of modernTCN')
parser.add_argument('--batch_size', type=int, default=50, help='batch size')
parser.add_argument('--ratio', type=int, default=2, help='ffn ratio')
parser.add_argument('--ls', type=int, default=51, help='large kernel sizes')
parser.add_argument('--ss', type=int, default=5, help='small kernel size')
parser.add_argument('--dp', type=float, default=0.5, help='drop out')
parser.add_argument('--hard', type=int, default=80, help='hard ratio (%) for mask')
parser.add_argument('--rg', type=int, default=1, help='re-gen data epoch')
parser.add_argument('--max_epoch', type=int, default=400, help='max train epoch')
parser.add_argument('--mix_test', action='store_true', default=False, help='是否混入测试集训练')

# 对照、消融实验的一些参数
parser.add_argument('--learnable_emb', action='store_true', default=False, help='是否使用可学习的 emb（原 modernTCN）而非 WVE')
parser.add_argument('--model', type=str, default='modernTCN', help='backbone 模型选择')
parser.add_argument('--manual', action='store_true', default=False, help='是否手动构建交织，需要 learnable_emb')
parser.add_argument('--pri', action='store_true', default=False, help='是否使用 PRI 而非 TOA')

parser_args = parser.parse_args()

use_cuda = True
device = torch.device(f"cuda:{parser_args.cuda}" if (
    use_cuda and torch.cuda.is_available()) else "cpu")
# device = torch.device("cpu")
print("CUDA Available: ", torch.cuda.is_available(), 'use:', device)


TAG_LEN = 12
BATCH_SIZE = parser_args.batch_size
INPUT_CHANNELS = 5 
# TODO 结构重参数化
# kernel_size = 51  # 51 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35
POS_D = D = 128  # embedding 这里是通道上的建模，不是时间上的建模你的卷积只要合适就行，不要太大了，但都是属于参数，提升没有多少，关键是域适应方法的集成  你要知道是 64*1024*512 的参数量很大的哈哈
P = 1
S = 1
R = parser_args.ratio
NUM_LAYERS = parser_args.num_layers  # 不同的层数对应模型大小不同，参数量也不同，要注意
DROP_OUT = parser_args.dp # dropout 仅指分类头的。不能在主干。ref. https://arxiv.org/abs/1801.05134v1
HARD_RATIO = parser_args.hard
RE_GEN_DATA_EPOCH = parser_args.rg
MAX_TRAIN_EPOCH = parser_args.max_epoch

if parser_args.manual and parser_args.learnable_emb:
    from mix_data_pos_all_manual import *
elif parser_args.pri:
    from mix_data_pos_all_PRI import *
else:
    from mix_data_pos_all import *

# TCN: modern TCN
# Linear 线性（模）值域宽尺度特征提取
# mask（mean 掩码）掩码难样本处理 randMask 随机掩码
# D L R: TCN 参数
# MT: 混入测试集 mix test
# Easy: 2,2,1,2
# Hr_: hard ratio + m r c (掩码方式)
NAME = f'{DATA_NAME}{HARD_RATIO}Hrr{RE_GEN_DATA_EPOCH}R'


IF_MIX_TEST = parser_args.mix_test
if IF_MIX_TEST:
    NAME = NAME + '_MT' 

IF_LAERNABLE_EMB = parser_args.learnable_emb
if IF_LAERNABLE_EMB:
    NAME = NAME + '_LEmb'
    from my_datastes import MyDataSet_woEmb as MyDataSet
else:
    from my_datastes import MyDataSet

from my_tools import *
seed_everything()

# %% 模型、优化器选择
if parser_args.model == 'modernTCN':
    NAME = f'TCN_{parser_args.ls}KS{parser_args.ss}_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
    # from TCNmodelPosAll import ModernTCN_DC
    from ModernTCN import ModernTCNnew
    # 不可结构重参数化：
    # model = ModernTCN_DC(INPUT_CHANNELS, WINDOW_SIZE, TAG_LEN, D=D,
    #                      P=P, S=S, kernel_size=kernel_size, r=R, num_layers=NUM_LAYERS, pos_D=POS_D).to(device)

    # 可结构重参数化：
    model = ModernTCNnew(INPUT_CHANNELS, 
                        TAG_LEN, 
                        D=D,
                        ffn_ratio=R, 
                        num_layers=NUM_LAYERS, 
                        large_sizes=parser_args.ls,
                        small_size=parser_args.ss,
                        backbone_dropout=0.,
                        head_dropout=DROP_OUT,
                        stem = IF_LAERNABLE_EMB
                        ).to(device)
    
    
    # * CNN 使用的优化器
    
    learn_rate = 4e-3
    # learn_rate = 2e-4
    # MIN_LR = 1e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)

    # < 50  e 0.004
    # < 100 e 0.002
    # < 150 e 0.001
    # < 200 e 0.0005
    # < 250 e 0.00025
    # < 300 e 0.000125
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50 // (BATCH_SIZE // 50), gamma=0.5)


elif parser_args.model == 'Transformer':
    if not IF_LAERNABLE_EMB:
        D = 128 * 5
        NAME = f'TF_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        from models.Transformer import Model, Configs
        configs = Configs()
        configs.d_model = D
        configs.e_layers = NUM_LAYERS
        configs.d_ff = D * R
        configs.n_heads = 4
        configs.dropout = DROP_OUT
        
        model = Model(configs=configs, wide_value_emb=True).to(device)
    
    else:
        D = 128 * 2
        NAME = f'TF_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
        from models.Transformer import Model, Configs
        configs = Configs()
        configs.d_model = D
        configs.e_layers = NUM_LAYERS
        configs.d_ff = D * R
        configs.n_heads = 2
        configs.dropout = DROP_OUT
        
        model = Model(configs=configs, wide_value_emb=False).to(device)
        
    # * TF 使用的优化器
    learn_rate = 0.
    optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 50 ** -1.5) 
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
elif parser_args.model == 'iTransformer':
    assert IF_LAERNABLE_EMB == True, "iTransformer 模型必须使用可学习的 emb. TODO"
    D = 128 * 2
    NAME = f'iTransformer_{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
    from models.iTransformer import Model, Configs
    configs = Configs()
    configs.e_layers = NUM_LAYERS
    configs.d_ff = configs.d_model * R
    configs.dropout = DROP_OUT
    
    model = Model(configs=configs, wide_value_emb=False).to(device)
        
    # * TF 使用的优化器
    learn_rate = 0.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learn_rate)
    lr_lambda = lambda step: (D ** -0.5) * min((step+1) ** -0.5, (step+1) * 50 ** -1.5) 
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
elif parser_args.model == 'TimesNet':
    assert IF_LAERNABLE_EMB == True, "TimesNet 模型必须使用可学习的 emb. TODO"
    
    D = 128
    NAME = f'TimesNet_{D}D{NUM_LAYERS}L{R}R{DROP_OUT*10:.0f}dp_{NAME}'
    from models.TimesNet import Model, Configs
    configs = Configs()
    configs.d_model = D
    configs.e_layers = NUM_LAYERS
    configs.d_ff = D * R
    configs.dropout = DROP_OUT
    
    
    model = Model(configs=configs, wide_value_emb=False).to(device)
    
    # 优化器
    learn_rate = 0.001
    optimizer = torch.optim.RAdam(model.parameters(), lr=learn_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
else:
    raise ValueError('model 选择错误')

# %%
# 训练数据准备
def make_data(d: List[List[np.ndarray]]):
    """生成数据集
    Args:
        d 数据集
    Returns:
        inputs2 = 特征
        targets = [[TAG] * 时间窗长度] * 样本数 one-hot 编码
    """
    inputs = np.array([i[0] for i in d], dtype=np.float32)
    targets = np.array([i[-1] for i in d], dtype=np.int64) - 1

    # 对 'TAG' 列的值 1~12 进行 one-hot 编码
    # targets = np.eye(12)[targets - 1] #(原标签 -1) 作为索引，生成对应的单位矩阵
    
    return torch.FloatTensor(inputs), torch.LongTensor(targets)


# %%

df_list, test_df_list = read_dfs()


def make_loader(batch_size, hard=1.):
    if IF_MIX_TEST:
        # 混入测试集 =====
        test_df_split_list: List[List[pd.DataFrame]] = [split_label(i) for i in test_df_list]
        d_train = mix_data_gen(df_list, 100, 100, 25, True, test_df_split_list)
        d_valid = mix_data_gen(df_list, 20, 50, 20, True, test_df_split_list)
        # 混入测试集 end =====
    else:
        d_train = mix_data_gen(df_list, 100, 100, 25, True)
        d_valid = mix_data_gen(df_list, 20, 50, 20, True)
    
    training_loader = Data.DataLoader(
        MyDataSet(*make_data(d_train), hard=hard), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validing_loader = Data.DataLoader(
        MyDataSet(*make_data(d_valid), hard=None), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    return training_loader, validing_loader

# %%
# 测试集 ==== 后面统一测试
test_windows = []

for test_df in test_df_list:
    # 通过时间窗切片的方法生成数据集
    for i in range(0, test_df.shape[0] - WINDOW_SIZE, WINDOW_SIZE):
        # 生成一个时间窗
        df_window = test_df.iloc[i:i+WINDOW_SIZE, :]
        # 生成一个样本
        m2, m3 = to_dict(df_window)
        # 加入到测试集中
        test_windows.append([m2, m3])

d_test = test_windows
testing_loader = Data.DataLoader(MyDataSet(*make_data(d_test)), batch_size=BATCH_SIZE, shuffle=True)
testing_loader_mini = Data.DataLoader(MyDataSet(*make_data(target_domain_data_gen(test_df_list[2], 20, 50))), batch_size=BATCH_SIZE, shuffle=True)


# %%
# 释放显卡内存
torch.cuda.empty_cache()

# %%
def criterion(outputs: torch.FloatTensor, targets: torch.FloatTensor):
    """自定义 loss function
    loss_1 : 模型最终输出的交叉熵损失
    """
    
    # loss_1 = focal_loss(outputs.view(-1, 12), targets.view(-1), alpha=None, reduction='mean')
    loss_1 = nn.CrossEntropyLoss()(outputs.view(-1, 12), targets.view(-1))
    return loss_1



# %%


def epoch_test_loss(model, validing_loader, testing_loader_mini):
    vaild_loss = 0
    with torch.no_grad():
        for data, target in validing_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            
            vaild_loss += criterion(output, target).item()
    vaild_loss /= len(validing_loader)

    # 注意这个 test loss mini
    test_loss = 0
    with torch.no_grad():
        test_loss = 0
        predictions = np.array([]).astype(int)
        targets = np.array([]).astype(int)
        for data, target in testing_loader_mini:
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            predictions = np.append(predictions, torch.argmax(
                output.view(-1, 12), -1).cpu().numpy())
            targets = np.append(targets, target.view(-1).cpu().numpy())

            test_loss += criterion(output, target).item()

        test_loss /= len(testing_loader_mini)
    return vaild_loss, test_loss, accuracy_score(predictions, targets)


# %%

loss_record = {"train": [], "vaild": [], "test": [], "acc": []}

epoch_start = 0 # 这里可以直接加载接着训练

# epoch_start = epoch
# epoch_start, loss_record = load_checkpoint(model, f'./saved_models/{NAME}_cp-{epoch_start}.pth', None, device)
# epoch_start, loss_record = load_checkpoint(model, f'saved_models/TCN_51KS5_128D24L2R0dp_PosAll80Hr1R_MT_cp-449.pth', None, device)
# epoch_start, loss_record = load_checkpoint(model, f'saved_models/TCN_51KS5_128D24L2R0dp_2lr_PosAll75Hr1R_MT_mloss_0.129_cp-822.pth', None, device)
# epoch_start, loss_record = load_checkpoint(model, f'saved_models/TCN_linear_mask_128D24L2R_POS_ALL_MT_mloss_0.17166166976094246_cp-295.pth', None, device)
# _, _ = load_checkpoint(model, f'./saved_models/ModernTCN_DC_modmix_cp-221.pth')

# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 25)

# test_loss_min = .3 if IF_MIX_TEST else .5
test_loss_min = 2.
# training_loader, validing_loader = make_loader(BATCH_SIZE)
print("参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad), end=' ')
print(NAME, "start train:", epoch_start)

# %%

# 自动、混合精度
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

model.to(device)
model.train()

l = tqdm(range(epoch_start, epoch_start + MAX_TRAIN_EPOCH))
for epoch in l:  # 在数据集上循环多次
    running_loss = 0.0
    # 重生成数据集
    if epoch % RE_GEN_DATA_EPOCH == 0:
        # training_loader, validing_loader = make_loader(BATCH_SIZE, hard=min(epoch / 100, 0.8))
        training_loader, validing_loader = make_loader(BATCH_SIZE, hard=HARD_RATIO * 0.01)
        

    # 先测试一下
    model.eval()
    v, t, a = epoch_test_loss(model, validing_loader, testing_loader_mini)
    loss_record["vaild"].append(v)
    loss_record["test"].append(t)
    loss_record["acc"].append(a)
    plot_loss(loss_record, f'{NAME}_{now}')
    
    
    # 保存最小的 test loss 模型
    if test_loss_min > loss_record["test"][-1]:
        test_loss_min = loss_record["test"][-1]
        print(f'Epoch {epoch} test loss min {test_loss_min}, acc {loss_record["acc"][-1]}')
        # save_checkpoint(epoch, loss_record, model, optimizer, f'./saved_models/{NAME}_mloss_{test_loss_min:.3f}_cp-{epoch-1}.pth')
        save_checkpoint(epoch, loss_record, model, optimizer, f'./saved_models/{NAME}_{now}_mloss.pth')
        # plot_loss(loss_record, f'{NAME}_{now}_mloss')

        
    model.train()
    for i, data in enumerate(training_loader):
        # 获取输入；数据是 [输入、标签] 的列表
        inputs, labels = data[0].to(device), data[-1].to(device)


        # 将参数梯度归零
        optimizer.zero_grad()

        with autocast():
            # forward + backward + optimize
            outputs = model(inputs)
            # 计算最后维度的交叉熵损失然后平均
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # loss.backward()
        # optimizer.step()

        running_loss += loss.item()
        l.set_description("E%d, loss=%.3f, vaild=%.3f, test=%.3f, acc=%.2f" %
                          (epoch + 1, loss.item(), loss_record["vaild"][-1], loss_record["test"][-1], loss_record["acc"][-1]))
        
    # if epoch < 300:
    lr_scheduler.step()
        
        
    loss_record["train"].append(running_loss / len(training_loader))
    
    if epoch % 50 == 49:
        save_checkpoint(epoch, loss_record, model, optimizer, f'./saved_models/{NAME}_{now}.pth')

print('Finished Training')
save_checkpoint(epoch, loss_record, model, optimizer, f'./saved_models/{NAME}_{now}_cp-{epoch}.pth')


# %%
plot_loss(loss_record, f'{NAME}_{now}')


print(f'learn_rate={learn_rate} batch_size={BATCH_SIZE}')
print("训练集样本数：", len(training_loader.dataset), "验证集样本数：", len(validing_loader.dataset))
print("训练集批次数：", len(training_loader), "验证集批次数：", len(validing_loader))
print("训练集损失：", loss_record["train"][-1], "验证集损失：", loss_record["vaild"][-1], "测试集损失：", loss_record["test"][-1])
print("模型参数量：", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("模型信息：", model)
# print('step lr last = 2e-4 cosin')
print("优化器信息：", optimizer)

# %%
# model=ModernTCN_DC(input_channels,seq_length,n_classes, D=D,P=P, S=S,kernel_size=kernel_size,num_layers=num_layers)
# model.to(device)
#
checkpoint = torch.load(f'./saved_models/{NAME}_{now}_mloss.pth')
#model = torch.load('./saved_models/tf_ModernTCN_DC_1000_cp-20.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

with torch.no_grad():
    for l in [validing_loader, 
              testing_loader, 
                # Data.DataLoader(MyDataSet(*make_data(target_domain_data_gen(test_df_list[0], 10, 100))), batch_size=BATCH_SIZE, shuffle=True),
                # Data.DataLoader(MyDataSet(*make_data(target_domain_data_gen(test_df_list[1], 10, 100))), batch_size=BATCH_SIZE, shuffle=True),
                Data.DataLoader(MyDataSet(*make_data(target_domain_data_gen(test_df_list[2], 10, 100))), batch_size=BATCH_SIZE, shuffle=True)
              ]:
        loss = 0
        predictions = np.array([]).astype(int)
        targets = np.array([]).astype(int)
        for data, target in tqdm(l):
            data = data.to(device)
            target = target.to(device)
            output = model(data)

            # 计算预测值
            predictions = np.append(predictions, torch.argmax(
                output.view(-1, 12), -1).cpu().numpy())

            targets = np.append(targets, target.view(-1).cpu().numpy())

            loss += criterion(output, target).item()

        loss /= len(l)

        confusion_matrix(predictions, targets, f'{NAME}_{now}_mloss{loss:.3}')
        print(f'Average loss: {loss}')



# %%
# 释放显卡内存
torch.cuda.empty_cache()


