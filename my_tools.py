import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
import torch
# %matplotlib widget
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


def seed_everything(seed: int = 3407):
    """ref. torch.manual_seed(3407) is all you need"""
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'set all seed: {seed}')


def TSNE_visualization(s_feature: np.ndarray, s_labels: np.ndarray, t_feature: np.ndarray, t_labels: np.ndarray, tag_len: int = 12):
    """TSNE 可视化域间的分布
    输入都是 2 维 numpy 数组，维度 1: 样本，维度 2: 特征
    s: 源域
    t: 目标域"""
    tsne = TSNE(n_components=2)

    # 拼接
    feature = np.concatenate((s_feature, t_feature), axis=0)

    # TSNE 降维
    feature = tsne.fit_transform(feature)
    cut = s_feature.shape[0]
    s_feature = feature[:cut, :]
    t_feature = feature[cut:, :]

    # 绘制 源域和目标域的散点图
    plt.figure()
    plt.scatter(feature[:cut, 0], feature[:cut, 1],
                c='r', label='源域', s=1, alpha=0.2)
    plt.scatter(feature[cut:, 0], feature[cut:, 1],
                c='b', label='目标域', s=1, alpha=0.2)
    plt.legend()
    plt.plot()

    # Plot the data points, 根据标签上色
    s_feature = feature[:cut, :]
    plt.figure()
    plt.scatter(feature[cut:, 0], feature[cut:, 1],
                c='gray', label='目标域', s=1, alpha=0.1)
    for i in range(0, tag_len):
        plt.scatter(s_feature[s_labels == i, 0],
                    s_feature[s_labels == i, 1], label=i+1, s=1, alpha=0.7)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('源域 t-SNE Visualization')
    plt.legend()
    plt.show()

    # Plot the data points, 根据标签上色
    plt.figure()
    plt.scatter(feature[:cut, 0], feature[:cut, 1],
                c='gray', label='源域', s=1, alpha=0.1)
    for i in range(0, tag_len):
        plt.scatter(t_feature[t_labels == i, 0],
                    t_feature[t_labels == i, 1], label=i+1, s=1, alpha=0.7)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('目标域 t-SNE Visualization')
    plt.legend()
    plt.show()


def save_checkpoint(epoch, loss_record, model: torch.nn.Module, optimizer: torch.optim.Optimizer, path):
    """保存模型 checkpoint"""
    state = {
        'epoch': epoch,
        'loss_record': loss_record,
        'model': model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)


def load_checkpoint(model, path, optimizer: torch.optim.Optimizer = None, device='cuda') -> Tuple[int, Dict[str, List[float]]]:
    """加载 checkpoint"""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])
    loss_record = state['loss_record']
    return state['epoch'], loss_record


def accuracy(predictions: np.ndarray, targets: np.ndarray):
    """计算硬标签的准确率"""
    return np.mean(predictions == targets)


def acc_logit(logits: torch.Tensor, targets: torch.Tensor):
    """计算 logits 的准确率"""
    predictions = np.argmax(
        logits.view(-1, 12).detach().cpu().numpy(), axis=1) + 1
    targets = np.argmax(
        targets.view(-1, 12).detach().cpu().numpy(), axis=1) + 1
    return accuracy(predictions, targets)


def confusion_matrix(predictions, targets, plot_name: str = None, tag_len: int = 12, average: str = 'weighted', if_save = True):
    """
    混淆矩阵
    """
    confusion_matrix = np.zeros((tag_len, tag_len))

    for target, prediction in zip(targets, predictions):
        confusion_matrix[target, prediction] += 1

    # 计算精确率、召回率、F1
    # 注：average='macro' 意味着每个类别的权重相同
    acc = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average=average)
    recall = recall_score(targets, predictions, average=average)
    f1 = f1_score(targets, predictions, average=average)
    log = f'Acc: {acc*100:.3f}, Pre: {precision*100:.3f}, Rec: {recall*100:.3f}, F1: {f1*100:.3f} [{average}]'
    print(plot_name)
    print(log)
    
    if plot_name:
        plt.figure()
        plt.imshow(confusion_matrix / np.maximum(1, np.sum(confusion_matrix, axis=1)[:, None]))
        # 同时在方格内显示数值
        for i in range(tag_len):
            for j in range(tag_len):
                plt.text(j, i, f'{confusion_matrix[i, j]:.0f}',
                         ha='center', va='center', color='blue')
        plt.title('confusion matrix')
        plt.xlabel('prediction')
        plt.ylabel('target')
        plt.colorbar()
        plt.title(f'{plot_name}\n{log}')
        if if_save:
            plt.savefig(f'saved_figs/{plot_name}.png')
        plt.show()

    return acc


def plot_loss(loss_record: Dict[str, List[float]], plot_name: str):
    plt.figure()
    plt.plot(loss_record["train"], label="train", linestyle="-", marker=".", linewidth=1, alpha=0.6)
    plt.plot(loss_record["vaild"], label="vaild", linestyle="-", marker=".", linewidth=1, alpha=0.6)
    plt.plot(loss_record["test"], label="test", alpha=0.9)
    plt.plot(loss_record["acc"], label="acc", alpha=0.9)
    plt.grid()
    plt.ylim(0, 3)
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title(f'{plot_name}\nmax acc: {max(loss_record["acc"]):.3f} min loss:{min(loss_record["test"]):.5f}')
    plt.savefig(f'saved_figs/{plot_name}_loss.png')
    plt.show()
    plt.close()


def load_pretrained_params(model: torch.nn.Module, path='./my_models/tf_s_2time5class_1000_minloss_cp-941.pth'):
    """加载预训练参数，只加载同名的层"""
    pretrained_dict = torch.load(path)['model_state_dict']
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(pretrained_dict, strict=False)
    return model
