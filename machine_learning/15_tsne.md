# 15. t-分布随机近邻嵌入 (t-SNE)

## 1. 核心原理

t-SNE（t-Distributed Stochastic Neighbor Embedding）是一种用于高维数据可视化和非线性降维的流形学习算法。其基本思路是：

1. **高维空间**：利用高斯分布将数据点之间的距离转化为相似度条件概率 $p_{j\vert{}i}$。

* $p_{j\vert{}i}$: 在高维空间中，以数据点 $x_i$ 为中心建立高斯分布时，选择 $x_j$ 作为其近邻的条件概率

2. **低维空间**：利用自由度为 1 的 $t$-分布（即柯西分布）计算低维空间中点的相似度条件概率 $q_{ij}$。选用 $t$-分布的长尾特性可以有效解决高维到低维的“拥挤问题”（Crowding Problem）。

* $q_{ij}$: 低维空间中数据点 $y_i$ 和 $y_j$ 之间的联合概率/相似度

3. **优化**：使用相对熵 / KL散度（Kullback-Leibler Divergence）来衡量高维分布与低维分布之间的差异，并通过梯度下降优化低维点的坐标。

## 2. 算法与数学公式

### 2.1 高维相似度 (高斯分布)

$$p_{j\vert{}i} = \frac{\exp(-\Vert{}\mathbf{x}_i - \mathbf{x}_j\Vert{}^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\Vert{}\mathbf{x}_i - \mathbf{x}_k\Vert{}^2 / 2\sigma_i^2)}, \quad p_{i\vert{}i} = 0$$

* $p_{j\vert{}i}$: 高维空间中数据点 $\mathbf{x}_i$ 选择 $\mathbf{x}_j$ 作为其近邻的条件概率
* $\mathbf{x}_i$: 第 $i$ 个高维数据点向量
* $\mathbf{x}_j$: 第 $j$ 个高维数据点向量
* $\mathbf{x}_k$: 高维空间中作为求和索引的第 $k$ 个数据点向量
* $\Vert{}\mathbf{x}_i - \mathbf{x}_j\Vert{}$: 数据点 $\mathbf{x}_i$ 与 $\mathbf{x}_j$ 之间的欧氏距离
* $\sigma_i$(西格玛): 以数据点 $\mathbf{x}_i$ 为中心的高斯分布的标准差（由困惑度 Perplexity 决定）
* $p_{i\vert{}i}$: 数据点自身对自身的条件概率（固定设为 0）

为了对称化：

$$p_{ij} = \frac{p_{j\vert{}i} + p_{i\vert{}j}}{2N}$$

* $p_{ij}$: 对称化后的高维空间数据点 $\mathbf{x}_i$ 与 $\mathbf{x}_j$ 之间的联合概率
* $p_{j\vert{}i}$: 以 $\mathbf{x}_i$ 为中心的条件概率
* $p_{i\vert{}j}$: 以 $\mathbf{x}_j$ 为中心的条件概率
* $N$: 数据集中数据点的总数量

### 2.2 低维相似度 (t-分布)

$$q_{ij} = \frac{(1 + \Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \Vert{}\mathbf{y}_k - \mathbf{y}_l\Vert{}^2)^{-1}}, \quad q_{ii} = 0$$

* $q_{ij}$: 低维空间中映射点 $\mathbf{y}_i$ 与 $\mathbf{y}_j$ 之间的联合概率
* $\mathbf{y}_i$: 高维数据点 $\mathbf{x}_i$ 映射到低维空间后的坐标向量
* $\mathbf{y}_j$: 高维数据点 $\mathbf{x}_j$ 映射到低维空间后的坐标向量
* $\mathbf{y}_k$: 低维空间中作为第一层外循环索引的第 $k$ 个坐标向量
* $\mathbf{y}_l$: 低维空间中作为第二层内循环索引的第 $l$ 个坐标向量
* $\Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}$: 映射点 $\mathbf{y}_i$ 与 $\mathbf{y}_j$ 之间的欧氏距离
* $q_{ii}$: 低维空间中数据点自身对自身的联合概率（固定设为 0）

### 2.3 目标函数 (KL 散度)

$$L = \text{KL}(P \Vert{} Q) = \sum_{i} \sum_{j \neq i} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

* $L$: 损失函数（Loss Function），即 KL 散度总和
* $\text{KL}(P \Vert{} Q)$: 描述高维概率分布 $P$ 与低维概率分布 $Q$ 之间差异的相对熵
* $P$: 所有高维联合概率 $p_{ij}$ 组成的概率分布矩阵
* $Q$: 所有低维联合概率 $q_{ij}$ 组成的概率分布矩阵
* $p_{ij}$: 高维空间中点 $i$ 和点 $j$ 的联合概率
* $q_{ij}$: 低维空间中点 $i$ 和点 $j$ 的联合概率
* $i$: 外层循环索引，遍历每个数据点
* $j$: 内层循环索引，遍历除 $i$ 以外的其他数据点

### 2.4 梯度计算

$$\frac{\partial L}{\partial \mathbf{y}_i} = 4 \sum_{j} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}^2)^{-1}$$

* $\frac{\partial L}{\partial \mathbf{y}_i}$: 损失函数 $L$ 对低维坐标向量 $\mathbf{y}_i$ 的偏导数（梯度）
* $L$: KL 散度目标函数
* $\mathbf{y}_i$: 当前需要计算梯度的低维数据点坐标向量
* $\mathbf{y}_j$: 其他低维数据点坐标向量
* $p_{ij}$: 高维空间中点 $i$ 与点 $j$ 的联合概率
* $q_{ij}$: 低维空间中点 $i$ 与点 $j$ 的联合概率
* $\Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}$: 低维点 $\mathbf{y}_i$ 与 $\mathbf{y}_j$ 之间的欧氏距离

## 3. ASCII 流程框架图

```
+------------------------------------+
| 高维空间点 X_i                     |
| 用高斯分布计算距离条件概率 p_ij      |
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| 初始化低维空间点 Y_i (通常为二维)   |
| 用 t-分布计算距离条件概率 q_ij      |
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| 计算损失：KL(P || Q)               |
+-----------------+------------------+
                  |
                  v
+------------------------------------+
| 计算梯度 dL/dY 并使用梯度下降/动量  |
| 持续更新 low-dim 坐标 Y_i           |
+------------------------------------+

```

## 4. Scikit-Learn 代码实现

```python
import numpy as np
from sklearn.manifold import TSNE

# 生成合成高维数据
np.random.seed(42)
X = np.random.randn(100, 50)  # 100个样本，50维特征

# 初始化与拟合 t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30.0,
    learning_rate='auto',
    n_iter=1000,
    random_state=42
)

X_embedded = tsne.fit_transform(X)

print("Original shape:", X.shape)
print("t-SNE Embedded shape:", X_embedded.shape)
print("First 3 embedded points:\n", X_embedded[:3])

```