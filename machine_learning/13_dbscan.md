# 13. DBSCAN (基于密度的聚类算法)

## 1. 核心原理

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的空间聚类算法。它将簇定义为密度相连的点构成的最大集合，能够发现任意形状的簇，并能有效识别噪声点。

两个关键超参数：

* $\epsilon$ (eps)：邻域半径。
* $MinPts$：给定邻域内所需的最少数据点数。

点分类：

1. **核心点（Core Point）**：在 $\epsilon$ 半径内包含至少 $MinPts$ 个点。
2. **边界点（Border Point）**：在 $\epsilon$ 半径内的点数少于 $MinPts$，但在某个核心点的 $\epsilon$ 邻域内。
3. **噪声点（Noise Point）**：既不是核心点也不是边界点的点。

## 2. 算法与数学公式

### 2.1 $\epsilon$-邻域

对数据点 $\mathbf{p} \in D$，其 $\epsilon$-邻域定义为：

$$N_\epsilon(\mathbf{p}) = \{ \mathbf{q} \in D \mid \text{dist}(\mathbf{p}, \mathbf{q}) \le \epsilon \}$$

* $N_\epsilon(\mathbf{p})$: 数据点 $\mathbf{p}$ 的 $\epsilon$-邻域集合
* $\epsilon(艾普西隆)$: 邻域半径超参数
* $\mathbf{p}$: 数据集中的指定数据点
* $D$: 整个数据集
* $\mathbf{q}$: 数据集中任意待判定的数据点
* $\text{dist}(\mathbf{p}, \mathbf{q})$: 数据点 $\mathbf{p}$ 与数据点 $\mathbf{q}$ 之间的距离度量（如欧氏距离）

### 2.2 密度相关概念

* **核心点判定**: $\vert{}N_\epsilon(\mathbf{p})\vert{} \ge MinPts$

* $\vert{}N_\epsilon(\mathbf{p})\vert{}$: 数据点 $\mathbf{p}$ 的 $\epsilon$-邻域内包含的数据点数量
* $\epsilon(艾普西隆)$: 邻域半径超参数
* $\mathbf{p}$: 待认定的数据点
* $MinPts$: 形成核心点所需的最小数据点数量超参数

* **直接密度可达 (Directly Density-Reachable)**: 若 $\mathbf{q} \in N_\epsilon(\mathbf{p})$ 且 $\mathbf{p}$ 是核心点，则 $\mathbf{q}$ 从 $\mathbf{p}$ 直接密度可达。

* $\mathbf{q}$: 目标数据点
* $N_\epsilon(\mathbf{p})$: 数据点 $\mathbf{p}$ 的 $\epsilon$-邻域集合
* $\epsilon(艾普西隆)$: 邻域半径超参数
* $\mathbf{p}$: 核心数据点

* **密度可达 (Density-Reachable)**: 存在序列 $\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n$，其中 $\mathbf{p}_1 = \mathbf{p}, \mathbf{p}_n = \mathbf{q}$，满足 $\mathbf{p}_{i+1}$ 从 $\mathbf{p}_i$ 直接密度可达。

* $\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n$: 密度可达路径上的点序列
* $\mathbf{p}$: 起始核心点
* $\mathbf{q}$: 终点数据点
* $\mathbf{p}_i$: 序列中第 $i$ 个数据点
* $\mathbf{p}_{i+1}$: 序列中第 $i+1$ 个数据点

* **密度相连 (Density-Connected)**: 存在点 $\mathbf{o}$，使得 $\mathbf{p}$ 和 $\mathbf{q}$ 均从 $\mathbf{o}$ 密度可达。

* $\mathbf{o}$: 作为中间桥梁的数据点
* $\mathbf{p}$: 目标数据点1
* $\mathbf{q}$: 目标数据点2

## 3. ASCII 流程框架图

```
                     +-----------------------+
                     |  遍历所有未访问的数据点 |
                     +-----------+-----------+
                                 |
                                 v
                     /-----------------------\\
                    <  |N_eps(p)| >= MinPts ? >
                     \\-----------------------/
                        /                 \\
                      Yes                 No
                      /                     \\
                     v                       v
          +--------------------+    +--------------------+
          |  标记 p 为核心点    |    |  标记 p 为噪声点   |
          |  创建新的簇 C      |    |  (后续可能变为边界点)|
          +---------+----------+    +--------------------+
                    |
                    v
          +--------------------+
          | 将邻域内点加入队列 |
          | 广度优先扩展簇 C    |
          +--------------------+


```

## 4. NumPy 纯代码实现

```python
import numpy as np

class DBSCANFromScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        n_samples = X.shape[0]
        labels = np.full(n_samples, -1)  # -1 表示噪声点
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        # 预先计算欧氏距离矩阵
        dist_matrix = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)

        for i in range(n_samples):
            if visited[i]:
                continue
            visited[i] = True

            neighbors = np.where(dist_matrix[i] <= self.eps)[0]

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 暂定为噪声
            else:
                # 扩展簇
                labels[i] = cluster_id
                seeds = list(neighbors)
                seeds.remove(i)

                while len(seeds) > 0:
                    curr_p = seeds.pop(0)
                    if not visited[curr_p]:
                        visited[curr_p] = True
                        curr_neighbors = np.where(dist_matrix[curr_p] <= self.eps)[0]
                        if len(curr_neighbors) >= self.min_samples:
                            seeds.extend(curr_neighbors)

                    if labels[curr_p] == -1:
                        labels[curr_p] = cluster_id

                cluster_id += 1

        return labels

if __name__ == "__main__":
    X = np.array([
        [1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]
    ])
    dbscan = DBSCANFromScratch(eps=3.0, min_samples=2)
    print("DBSCAN Cluster Labels:", dbscan.fit_predict(X))


```