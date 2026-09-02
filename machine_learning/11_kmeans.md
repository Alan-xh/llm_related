# 11. k-均值聚类 (k-Means Clustering)

## 1. 核心原理

k-均值聚类是一种无监督学习算法，用于将未标记的数据集划分为 $k$ 个不同的簇（Clusters）。其核心原理是：

1. 随机指定 $k$ 个初始聚类中心（Centroids）。
2. 将每个数据点分配给距离其最近的聚类中心。
3. 根据分配给每个簇的数据点更新聚类中心（计算均值）。
4. 重复步骤2与3，直到聚类中心不再显著改变或达到最大迭代次数（收敛）。

* k: 聚类簇的数量/设定的类别数

目标是最小化所有簇内数据点到对应簇中心的平方距离之和（即簇内平方和 / WCSS）。

## 2. 算法与数学公式

### 2.1 距离度量

常用欧几里得距离（Euclidean Distance）度量数据点 $\mathbf{x}_i$ 与聚类中心 $\boldsymbol{\mu}_j$ 之间的距离：


$$d(\mathbf{x}_i, \boldsymbol{\mu}_j) = \Vert{}\mathbf{x}_i - \boldsymbol{\mu}_j\Vert{}_2 = \sqrt{\sum_{d=1}^{D} (x_{id} - \mu_{jd})^2}$$

* d(·): 距离函数/距离测度
* $\mathbf{x}_i$: 第 $i$ 个数据点（向量）
* $\boldsymbol{\mu}$(缪)_j: 第 $j$ 个聚类中心的坐标向量
* $\Vert{}\cdot{}\Vert{}_2$: L2范数（欧几里得范数 / 欧氏距离）
* D: 特征维度/数据的空间维度数
* d: 特征维度的索引变量
* $x_{id}$: 第 $i$ 个数据点在第 $d$ 个维度上的特征值
* $\mu$(缪)_{jd}: 第 $j$ 个聚类中心在第 $d$ 个维度上的特征值

### 2.2 目标函数 (Inertia / WCSS)

$$J = \sum_{j=1}^{k} \sum_{i \in C_j} \Vert{}\mathbf{x}_i - \boldsymbol{\mu}_j\Vert{}^2$$


其中 $C_j$ 表示第 $j$ 个簇的数据点集合，$\boldsymbol{\mu}_j$ 是该簇的中心点。

* J: 目标函数值（簇内平方和 / WCSS / Inertia）
* k: 聚类簇的总数
* j: 簇的索引变量（$j = 1, 2, \dots, k$）
* $C_j$: 第 $j$ 个簇包含的数据点集合
* i: 数据点的索引变量
* $\mathbf{x}_i$: 属于簇 $C_j$ 的第 $i$ 个数据点（向量）
* $\boldsymbol{\mu}$(缪)_j: 第 $j$ 个簇的中心点向量
* $\Vert{}\cdot{}\Vert{}^2$: 欧几里得距离的平方/L2范数的平方

### 2.3 质心更新公式

$$\boldsymbol{\mu}_j = \frac{1}{\vert{}C_j\vert{}} \sum_{i \in C_j} \mathbf{x}_i$$

* $\boldsymbol{\mu}$(缪)_j: 更新后的第 $j$ 个簇的质心（中心点向量）
* $\vert{}C_j\vert{}$: 第 $j$ 个簇包含的数据点总数量（集合 $C_j$ 的势/大小）
* i: 数据点的索引变量
* $C_j$: 第 $j$ 个簇的数据点集合
* $\mathbf{x}_i$: 属于簇 $C_j$ 的第 $i$ 个数据点（向量）

## 3. ASCII 流程框架图

```

+-------------------------------------------------+
|               初始化 k 个质心                    |
+-------------------------------------------------+
|
v
+-------------------------------------------------+
|  计算距离：将每个点分配给最近的质心 (E-step)      |
+-------------------------------------------------+
|
v
+-------------------------------------------------+
|  更新质心：计算每个簇内所有点的均值 (M-step)     |
+-------------------------------------------------+
|
/-----------+-----------\
|                         |
[质心变动 > 阈值]         [质心不再变化/收敛]
|                         |
\-----------<-------------/
|
v
+-------------------------+
|        完成聚类         |
+-------------------------+


```

## 4. NumPy 纯代码实现

```python
import numpy as np

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        
    def fit(self, X):
        n_samples, n_features = X.shape
        # 1. 随机选择 k 个样本作为初始质心
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]
        
        for iteration in range(self.max_iters):
            # 2. 计算每个样本到各个质心的距离 (n_samples, k)
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
            
            # 分配聚类标签
            labels = np.argmin(distances, axis=1)
            
            # 3. 更新质心
            new_centroids = np.zeros((self.k, n_features))
            for j in range(self.k):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroids[j] = np.mean(cluster_points, axis=0)
                else:
                    new_centroids[j] = X[np.random.choice(n_samples)]
            
            # 检查收敛条件
            centroid_shift = np.sum(np.linalg.norm(new_centroids - self.centroids, axis=1))
            self.centroids = new_centroids
            if centroid_shift < self.tol:
                break
                
        return labels

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
        np.random.normal(loc=[5, 5], scale=0.5, size=(50, 2)),
        np.random.normal(loc=[0, 5], scale=0.5, size=(50, 2))
    ])
    
    kmeans = KMeansFromScratch(k=3)
    labels = kmeans.fit(X)
    print("Clusters assigned, counts per cluster:", np.bincount(labels))


```