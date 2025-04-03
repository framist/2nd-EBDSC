![logo of Wide-Value-Embs TCN](asserts/image.png)


*本归档为第二届“火眼金睛”电磁大数据非凡挑战赛（2nd EBDSC）金奖作品。*

> [!WARNING]
> 目前仅包含主干训练代码，不包括数据集、一些后处理、模型评估等代码。
> 
> **本仓库已归档**。更进一步的说明与使用，请关注我们未来的工作：
>
> [WVEmbs with its Masking: A Method For Radar Signal Sorting](http://arxiv.org/abs/2503.13480)
>
> ...

# 方法

## Wide-Value-Embs TCN - Proposed Method

### Wide-value-embeddings Generation for Interleaved Signals

针对宽值域尺度交织信号的嵌入生成

设长度为 $L$ 的 $N$ 维变量 $`{\bf s}_{\text{PDW}} \in \mathbb {R}^{L \times N}`$ 是输入的信号，$`D`$ 是所需的嵌入的维度，$`E_{i} \in \mathbb{R}^{D}`$ 是嵌入，这由所使用的 backbone 网络的输入决定。则我们可以设置步长 $d_\mathrm{step} \in \mathbb{N}^+$ （以下简写为 $d$）与 缩放系数 $k \in \mathbb{N}^+$ 来控制最大的模数上界 $M = k^{D/d}$。反之，我们也可以由 $M,k$ 决定 $d_\mathrm{step} =  D \log_{M}{k}$。

$M$ 是最大的模数上界，最大的模数是 $m_{\mathrm{max}} = M / k^{1}$, 意味输入数据值的模 $m_{\mathrm{max}}$ 无信息损失，这里我们默认 $1$ 是最小的模，因此我们需要线性变化输入数据以使得数据的信息能体现在模 $1 \sim m_{\mathrm{max}}$ 的**剩余系**中。

于是我们通过下式子得到宽值域嵌入 ${\bf E}_{\text{PDW}} \in \mathbb {R}^{L \times N \times D}$ ：


```math
{\bf E}_{\text{PDW}}^{l,n,d i+j} =
f_{\omega =1}(\frac{{{\bf s}^{l,n}_{\text{PDW}}}}{M^{d i/D}} + \frac{j}{d}) 
,\quad
i \in \{0,1,...,D/d -1\} ,\quad j\in \{1,2,...,d\}
```


其中 $di+j\in \{1,2,...,D\}$ 是嵌入的一个维度，$x$ 是该输入 token 的值。


$f_{\omega =1}(\cdot)$ 是所使用的周期为 1 的线性周期函数，我们所使用的是

$$
f_{\omega =1}(x) \doteq x \bmod 1 \cdot 2-1
$$

也就是说，宽值域嵌入的相当与一个正整数倍的线性周期函数族，波长形成了从$1$到$M$的正整数 $`k`$ 倍，这样可以防止在之后的掩码中产生信息泄露，即只有在更大 $`m`$ 的维度才包含小尺度下的信息。我们选择了这个函数，因为我们假设在卷积运算中，它可以使模型轻松学习相对的关注，因为对于任何固定的值 $x$ 偏移 $k$，$`{\bf E}(x)`$ 与 ${\bf E}(x+k)$ 差值不变。

**反向宽值域嵌入**

宽值域嵌入的逆变换是：

```math
\hat{\bf s}^{l,n}_{\text{PDW}} =
\sum_{i=0}^{D/d-1} 
\frac{\sum_{j=1}^{d}
{f^{-1}_{\omega =1}(
\hat{\bf E}}_{\text{PDW}}^{l,n,d i+j} ) - \frac{j}{d} }
{d} 
M^{\frac{di}{D} }
```

其中 $`f^{-1}_{\omega =1}`$ 是 $`f_{\omega =1}`$ 的在任意周期内的反函数。即我们使用的

```math
f^{-1}_{\omega =1}(x) = (x + 1)/ 2
```

### Hard Sample Mining (as Interleave Construction) Based on Masking

通过掩码构建交织，可以迫使主干神经网络通过序列信号深层的特征分类信号，而非肤浅的简单统计特征。
我们将掩码表示为 $\textbf{M}$，将掩码区域的填充值表示为 $\textbf{z}$. 掩码后的嵌入可以表示为 $\textbf{M}*\textbf{x} + (1-\textbf{M}) * \textbf{z}$接下来，我们开始按照我们的设计原则构建掩码策略。

#### Value Dimension 值域维度

由于我们的**宽值域嵌入**妥善设计，更高的嵌入维度 $d$ 意味着特征从有用到平凡的变化 $`f \to g`$，如下图所示，更低的维度的蕴含语义更复杂，甚至趋向于随机值，而高维度的特征非常明显，用简单的统计方法就能分类。因此，我们对嵌入的以下维度进行掩码

已知模 $`m`$ 查找进行掩码的最低维度函数 ${\mathrm d}_{\mathrm {low}}(\cdot)$

```math
\begin{align} 
{\mathrm d}_{\mathrm {low}}(m) = \left \lfloor d \cdot \log_{k} m \right \rfloor
\end{align}
```

这样我们对 $`d > {\mathrm d}_{\mathrm {low}}(m)`$ 的维度掩码，就能近似构建极大极小差值为 $`m`$ 信号的交织情况。
掩码函数 $`F_\mathrm{mask}(\cdot, d_{\mathrm{low}})`$ 输入为宽值域嵌入 $`{\bf E}_{\text{PDW}} \in \mathbb {R}^{L \times N \times D}`$ ，输出为 $`{\bf E}_{\text{PDW}}^{\mathrm{masked}} \in \mathbb {R}^{L \times N \times D}`$ 
我们有以下掩码方案：
设 $`E_{}=[ e_{1}^{l,n},e_{2}^{l,n},...,e_{D}^{l,n} ]`$，这里的 $`d_{\mathrm{low}}`$ 表示为 $`{\mathrm d}_{\mathrm {low}}(m)`$ 的一个随机采样

```math
\mathrm{F}_\mathrm{mask}({\bf E}, d_{\mathrm{low}}) =
\left\{
\begin{matrix} 
    \mathcal{R} \left( e_d \right), & d \ge d_{\mathrm{low}} \\
    e_d, & d < d_{\mathrm{low}}
\end{matrix}
\right.
```

其中，$`\mathcal{R}(\cdot)`$ 表示产生一个相同、元素均匀分布在 $[0,1]$ 之间的张量，对应一般的 `rand_like` 函数。

掩码遮蔽了平凡的特征 $g$，随机掩码能大大提升抗噪性能。


## 文件结构

训练评估模型主要文件：`tcn_pos_all.py`

---

# 竞赛信息

第二届“火眼金睛”电磁大数据非凡挑战赛（2nd EBDSC）

[主页](https://mjs.datacastle.cn/cmptDetail.html?id=847) | [说明](https://challenge.datacastle.cn/v3/cmptDetail.html?id=847) | [模型提交示例](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/%E6%A8%A1%E5%9E%8B%E6%8F%90%E4%BA%A4%E7%A4%BA%E4%BE%8B.html)

## 赛题描述

在电磁信号处理中，设备侦收的往往是来自多个辐射源的混叠数据，实现多目标混叠信号的准确提取是后续身份判识及威胁推测的先决条件。随着雷达技术体制不断发展，信号种类愈发复杂多变，传统基于经验规则的方法已无法满足复杂信号准确提取需求。本赛题旨在解决对多目标多类型混叠信号的分辨提取问题，通过给定的模板信号（训练样本），实现对测试场景数据中已知信号（若干模板信号）和未知信号(模板以外其他信号)的提取与标注。

## 数据说明

**数据特征**

1. **训练数据:** 设置雷达每种信号类型的<u>持续发射时间为 10S</u>，共包含<u>12 种信号类型</u>的样本数据，每种类型的样本数据存为 1 个 txt 文件，共 12 个.txt 文件，每个训练集文件由一系列时序数据点组成，每个数据点主要由六维特征参数描述，如下图所示，从左到右依次为时间戳 TOA、特征 A、特征 B、特征 C、特征 D、标签 ID。每个文件中只包含该类型无噪声纯净的单类信号（无混叠、丢失、错误及干扰）。

    > 全部时序信号一起输入，每个样本点打一个标签
    >
    > * 第一列 TOA (Time of Arrival)（单位 s）
    > * 第二列频率（单位 MHz） | 载波频率(Radio Frequency, RF)
    > * 第三列脉宽（单位 μs）| 脉冲宽度(Pulse Width, PW)，
    > * 第四列幅值（dB）| 脉冲幅度(Pulse Amplitude, PA)
    > * 第五列到达角 | 到达方向(Direction of Arrival, DOA)
    > * 第六列标签
    >
    > 除了瞬时参数，经多次测量或通过计算可得到的参数为二次参数，最主要的代表是脉冲重复间隔(Pulse Repetitive Interval, PRI)。


2. **验证数据:** 多种信号类型的混叠数据，共设置 3 种场景的验证数据，每种场景数据存为 1 个.txt 文件，数据格式与训练数据相同，提供真实标签 ID，选手可用于算法的验证改进。
    
4. **测试数据：** 若干已知及未知干扰信号的混叠数据，包含测量误差等情况，以更贴近实际场景，验证候选方法的鲁棒性，测试数据由组委会保留。其中已知信号的 ID 和训练集的 ID 保持一致，<u>未知信号的 ID 标注为 UN</u>。

4.**测试规定：** 测试数据不带标签列，输出结果顺序要与测试数据顺序保持一致。输出结果为单标签列 txt 文件。
测试文件输入路径和预测结果保存路径以参数形式给定，共有三个测试文件：Scene1.txt, Scene2.txt, Scene3.txt, 选手推理完成后需要生成 3 个结果文件, 保存在给定路径下， 三个输出文件的命名分别是 Scene1.txt, Scene2.txt, Scene3.txt, 与测试集一一对应。具体可参照示例 demo: run.py。选手需严格遵守上述规定，否则会导致作品测试无法成功。


## 提交说明

1. 线上提交分多轮进行
    第一轮提交时间为 2024 年 1 月 3 日 00:00-2024 年 1 月 5 日 24:00，开放提交期间每支团队每天最多成功提交 1 次。
    第二轮提交时间待定。
2. 点击链接查看 [模型提交示例](https://pu-datacastle.obs.cn-north-1.myhuaweicloud.com/%E6%A8%A1%E5%9E%8B%E6%8F%90%E4%BA%A4%E7%A4%BA%E4%BE%8B.html)
3. 提交 demo 详见 [https://caiyun.139.com/m/i?1E5C3MH8eXBqB](https://caiyun.139.com/m/i?1E5C3MH8eXBqB)  提取码:3F7u
4. 提交限制: 开放提交期间每支团队每天最多成功提交 1 次，提交模型大小限制: 500MB，推理时长限制: 1 小时

## 评分标准

**评价维度和指标**

1. 信号提取标注精确率：首先计算每种信号类型的精确率：该类型预测 ID 正确的脉冲数/预测为该类型的脉冲总数，然后所有类型的精确率取平均值。
2. 信号提取标注召回率：首选计算每种信号类型的召回率：该类型预测 ID 正确的脉冲数/该类型实际脉冲总数，然后所有类型的召回率取平均值。


---

by Framist - 「电磁利剑 101」战队 - 如有任何问题，请联系我们
