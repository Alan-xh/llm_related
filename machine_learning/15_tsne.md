
# 15. t-分布随机近邻嵌入 (t-SNE)

## 1. 核心原理

t-SNE（t-Distributed Stochastic Neighbor Embedding）是一种用于高维数据可视化和非线性降维的流形学习算法。其基本思路是：

1. **高维空间**：利用高斯分布将数据点之间的距离转化为相似度条件概率 $p_{j\vert{}i}$。
2. **低维空间**：利用自由度为 1 的 $t$-分布（即柯西分布）计算低维空间中点的相似度条件概率 $q_{ij}$。选用 $t$-分布的长尾特性可以有效解决高维到低维的“拥挤问题”（Crowding Problem）。
3. **优化**：使用相对熵 / KL散度（Kullback-Leibler Divergence）来衡量高维分布与低维分布之间的差异，并通过梯度下降优化低维点的坐标。

## 2. 算法与数学公式

### 2.1 高维相似度 (高斯分布)

$$p_{j\vert{}i} = \frac{\exp(-\Vert{}\mathbf{x}_i - \mathbf{x}_j\Vert{}^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\Vert{}\mathbf{x}_i - \mathbf{x}_k\Vert{}^2 / 2\sigma_i^2)}, \quad p_{i\vert{}i} = 0$$


为了对称化：


$$p_{ij} = \frac{p_{j\vert{}i} + p_{i\vert{}j}}{2N}$$

### 2.2 低维相似度 (t-分布)

$$q_{ij} = \frac{(1 + \Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \Vert{}\mathbf{y}_k - \mathbf{y}_l\Vert{}^2)^{-1}}, \quad q_{ii} = 0$$

### 2.3 目标函数 (KL 散度)

$$L = \text{KL}(P \Vert{} Q) = \sum_{i} \sum_{j \neq i} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

### 2.4 梯度计算

$$\frac{\partial L}{\partial \mathbf{y}_i} = 4 \sum_{j} (p_{ij} - q_{ij})(\mathbf{y}_i - \mathbf{y}_j)(1 + \Vert{}\mathbf{y}_i - \mathbf{y}_j\Vert{}^2)^{-1}$$

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

