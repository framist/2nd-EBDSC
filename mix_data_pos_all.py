# %% [markdown]
# # 造数据
# 模归一化变为「大小尺度特征提取」

# %%
import numpy as np
import pandas as pd
from typing import List, Tuple
from joblib import Parallel, delayed
import random


TAG_LEN = 12
WINDOW_SIZE = 1024
DROP_SIG_RATIO = 0.99
DATA_NAME = f'PosAll'


def read_txt(filepath):
    """读取 txt 文件，并以 DataFrame 返回
    优化：使用 panda.read_csv 读取数据集
    """
    # columns = ["TOA", "频率", "脉宽", "幅值", "到达角", "标签"]
    columns = ["TOA", "RF", "PW", "PA", "DOA", "TAG"]
    df = pd.read_csv(filepath, sep='\s+', names=columns)
    # 数据类型规范
    df["TOA"] = df["TOA"].astype(np.float32)
    df["RF"] = df["RF"].astype(np.float32)
    df["PW"] = df["PW"].astype(np.float32)
    df["PA"] = df["PA"].astype(np.float32)
    df["DOA"] = df["DOA"].astype(np.float32)
    df["TAG"] = df["TAG"].astype(np.int32)
    return df


# 并行读取训练数据集
def read_train_data(file_index):
    return read_txt(f"../训练数据集/信号类型{file_index+1}训练集.txt")


# 并行创建验证数据集
def read_test_data(file_index):
    return read_txt(f"../验证数据集/验证集{file_index+1}.txt")


def read_dfs() -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    """df_list, test_df_list = read_dfs()

    Returns:
        tuple[List[pd.DataFrame], List[pd.DataFrame]]: _description_
    """
    print("并行读取数据集中...", end=' ')
    df_list: List[pd.DataFrame] = Parallel(n_jobs=-1)(
        delayed(read_train_data)(i) for i in range(TAG_LEN)
    )
    test_df_list: List[pd.DataFrame] = Parallel(n_jobs=-1)(
        delayed(read_test_data)(i) for i in range(3)
    )
    print("读取数据集 end")
    return df_list, test_df_list


def pre_reshape(df: pd.DataFrame, sample_ratio: float = None) -> pd.DataFrame:
    """数据时间尺度上变化，模拟频移
    注意：输入的 TAG 需为常数，即单个 TAG 数据
    [129720, 120110, 193320, 133054, 137488, 143848, 155580, 222870, 150702, 120392, 157934, 22112]
    df0 in, [0.000130, 9.997574], 共 9.997444
    df1 in, [0.500132, 10.499588], 共 9.999456
    df2 in, [0.000237, 9.719326], 共 9.719089
    df3 in, [0.000198, 9.996972], 共 9.996774
    df4 in, [1.300043, 11.297212], 共 9.997169
    df5 in, [0.800052, 10.799469], 共 9.999417
    df6 in, [0.500093, 10.499944], 共 9.999850
    df7 in, [3.000056, 7.997694], 共 4.997639
    df8 in, [6.000054, 10.999905], 共 4.999850
    df9 in, [6.000223, 10.022663], 共 4.022440
    df10 in, [4.000152, 7.592672], 共 3.592521
    df11 in, [0.000449, 9.999234], 共 9.998785
    """

    # df_scaled = df.copy()
    # # 随机截取其中 TOA 3.5s 窗口长的数据
    # cut_len = 3.5
    # start = np.random.uniform(df["TOA"].min(), df["TOA"].max() - cut_len)
    # df_scaled: pd.DataFrame = df[(df["TOA"] >= start) & (df["TOA"] <= start + cut_len)].copy()

    # 随机截取其中 TOA 5s 窗口长的数据
    cut_len = 5
    start = np.random.uniform(df["TOA"].min(), df["TOA"].max())
    df_scaled: pd.DataFrame = df[(df["TOA"] >= start) & (
        df["TOA"] <= start + cut_len)].copy()

    # 随机信号丢失
    if sample_ratio is None:
        df_scaled = df_scaled.sample(frac=np.random.uniform(DROP_SIG_RATIO, 1))
    else:
        df_scaled = df_scaled.sample(frac=sample_ratio)

    # 频移。
    df_scaled["TOA"] *= np.clip(np.random.normal(1, 0.2), 0.5, 2)

    # 去除原始的 TOA 偏移
    df_scaled["TOA"] -= df_scaled["TOA"].min()

    # 随机进入时移
    # df_scaled["TOA"] += np.random.uniform(0, cut_len)
    df_scaled["TOA"] += np.clip(np.random.exponential(1), 0, cut_len)

    return df_scaled


def to_dict(df: pd.DataFrame, reshap=False, noise=1., manual_interleave=False):
    """数据转换为字典索引
    - 除以的超参数意义为「数字化单位」    
    reshap 需要有真实的 "TAG" 列
    """

    def normalize(d: pd.Series) -> pd.Series:
        """标准化"""
        return (d - d.mean()) / d.std()
    
    def mod_normalize(d: pd.Series, mod: int) -> pd.Series:
        """模 标准化 -> [0, 1)"""
        return (d % mod) / mod

    df = df.copy()
    # df["PRI"] = df["TOA"].diff().fillna(0)

    d_2 = pd.DataFrame()
    d_tag = pd.DataFrame()

    # 抽取不同的 tag 变换
    if reshap:
        for i in range(TAG_LEN):  # 需要有真实的 "TAG" 列
            # RF 变换 (0.2, 2) 的缩放倍率参考 3 个测试集最大值
            di = df[df["TAG"] == i+1]["RF"]
            df.loc[df["TAG"] == i+1, "RF"] = np.random.uniform(0.2, 2) * noise * (di - di.mean()) + di.mean() + \
                np.random.uniform(0, 100) * noise
            # np.random.normal(0, 1, di.shape).astype(np.float32) + \

            # PW 变换 (0.5, 2) 的缩放倍率参考 3 个测试集最大值
            di = df[df["TAG"] == i+1]["PW"]
            df.loc[df["TAG"] == i+1, "PW"] = np.random.uniform(0.5, 2) * noise * (di - di.mean()) + di.mean() + \
                np.random.uniform(0, 10) * noise

            # DOA 变换
            df.loc[df["TAG"] == i+1, "DOA"] += np.random.uniform(0, 90) * noise

        # 整体偏移 需要
        df["RF"] += np.random.uniform(-1000, 1000) * noise

        df["PW"] += np.random.uniform(-10, 10) * noise

        df["DOA"] += np.random.uniform(-60, 60) * noise

    if manual_interleave:
        DOA_MOD = 90
        RF_MOD = 100
        PW_MOD = 10
        d_2["TOA"] = (df["TOA"] - df["TOA"].min())

        # * RF
        d_2["RF"] = mod_normalize(df["RF"], RF_MOD)

        # * PW
        d_2["PW"] = mod_normalize(df["PW"], PW_MOD)

        # * PA
        d_2["PA"] = normalize(df["PA"])

        # * DOA
        d_2["DOA"] = mod_normalize(df["DOA"], DOA_MOD)

        # * TAG
        d_tag["TAG"] = df["TAG"].astype(np.int32)
        return d_2.values, d_tag.values
    
    # * PRI
    # d_2["PRI"] = (df["PRI"] - 0.0002) / 0.0005

    # * POS
    # d_2["PRI"] = df["TOA"].diff().fillna(0) * 5e5
    d_2["TOA"] = (df["TOA"] - df["TOA"].min()) * 5e5

    # * RF
    d_2["RF"] = (df["RF"])

    # * PW
    d_2["PW"] = (df["PW"]) * 10

    # * PA < 0
    d_2["PA"] = (df["PA"])

    # * DOA
    d_2["DOA"] = (df["TOA"])

    # * TAG
    d_tag["TAG"] = df["TAG"].astype(np.int32)

    return d_2.values, d_tag.values


def split_label(df: pd.DataFrame) -> List[pd.DataFrame]:
    """分离混杂的 TAG 数据"""
    df_list = []
    for i in range(TAG_LEN):
        df_list.append(df[df["TAG"] == i+1].copy().reset_index(drop=True))
    return df_list


def _gen_mix_job(df_list: List[pd.DataFrame], size_2, if_time_reshap=False, test_df_split_list: List[List[pd.DataFrame]] = None, t=None, sample_ratio: float = None):
    """生成混合窗口

    Args:
        df_list (List[pd.DataFrame])
        size_2
        if_time_reshap (bool, optional): Defaults to False. 是否时间尺度变换
        test_df_split_list 非 None 则混合测试集

    """
    if t is None:
        t = range(TAG_LEN)
        # t = [0, 2, 5, 7]
        # t = random.sample(range(TAG_LEN), random.randint(9, TAG_LEN))

    if test_df_split_list:
        df_list_new = []
        # 对于每一个 t 随机选择
        for i in t:
            d = random.sample(
                [j for j in [k[i] for k in test_df_split_list] if j.shape[0] > 0] + [df_list[i]], 1)
            df_list_new.extend(d)
    else:
        df_list_new = df_list

    if if_time_reshap:
        df = pd.concat([pre_reshape(df_list_new[i], sample_ratio)
                       for i in t], ignore_index=True)
    else:
        df = pd.concat([df_list_new[i] for i in t], ignore_index=True)
    df = df.sort_values(by="TOA", ascending=True, ignore_index=True)

    mixed_windows = []
    for _ in range(size_2):
        start = random.randint(0, len(df) - WINDOW_SIZE)
        m2, m3 = to_dict(df[start:start + WINDOW_SIZE], reshap=True)
        mixed_windows.append([m2, m3])
    return mixed_windows


def mix_data_gen(df_list: List[pd.DataFrame], size_1: int, size_2: int, n_jobs: int = -1, if_time_reshap=False, test_df_split_list=None, t=None, sample_ratio: float = None):
    mix_windows = []

    # 并行生成混合窗口
    mixed_results = Parallel(n_jobs=n_jobs)(
        delayed(_gen_mix_job)(df_list, size_2, if_time_reshap, test_df_split_list, t, sample_ratio) for _ in range(size_1)
    )

    for result in mixed_results:
        mix_windows.extend(result)

    return mix_windows


def _target_domain_job(df: pd.DataFrame, size_2):
    mixed_windows = []
    for _ in range(size_2):
        start = random.randint(0, len(df) - WINDOW_SIZE)
        m2, m3 = to_dict(df[start:start + WINDOW_SIZE], reshap=False)
        mixed_windows.append([m2, m3])
    return mixed_windows


def target_domain_data_gen(df: pd.DataFrame, size_1: int, size_2: int, n_jobs: int = -1) -> List:
    """此时 to_dict 的 reshap: bool = False
    size_1 与 size_2 意义无区别，乘积为生成的数据量
    """
    mix_windows = []

    # 并行生成混合窗口
    mixed_results = Parallel(n_jobs=n_jobs)(
        delayed(_target_domain_job)(df, size_2) for _ in range(size_1)
    )

    for result in mixed_results:
        mix_windows.extend(result)

    return mix_windows
