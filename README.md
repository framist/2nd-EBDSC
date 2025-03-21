![logo of Wide-Value-Embs TCN](asserts/image.png)

# 2nd EBDSC


*本归档为第二届“火眼金睛”电磁大数据非凡挑战赛（2nd EBDSC）金奖作品。*

> [!WARNING]
> 目前仅包含主干训练代码，不包括数据集、一些后处理、模型评估等代码。
> 
> **本仓库已归档**。更进一步的说明与使用，请关注我们未来的工作：
>
> [WVEmbs with its Masking: A Method For Radar Signal Sorting](http://arxiv.org/abs/2503.13480)
>
> ...


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

## 模型训练


## 提交评估代码（归档）

---

by Framist - 「电磁利剑 101」战队 - 如有任何问题，请联系我们
