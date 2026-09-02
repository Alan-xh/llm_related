# K-Means++ 聚类算法

## 1. 算法原理

经典的 K-Means 算法对初始聚类中心的选择非常敏感，糟糕的初始质心可能导致算法收敛缓慢或陷入劣质局部最优解。

**K-Means++** 针对初始质心的选择做出了改进，其核心思想是：**初始聚类中心彼此之间的距离越远越好**。

初始化步骤如下：

1. 从数据集 $X$ 中随机均匀挑选一个样本作为第一个聚类中心 $\mu_1$。

* X: 输入的数据集
* μ(缪)_1: 选出的第 1 个聚类中心（质心）

2. 对于数据集中每一个样本 $x_i$，计算其与当前已选出的最近质心之间的最短距离 $D(x_i)$。

* x_i: 数据集中的第 i 个数据样本
* D(x_i): 样本 x_i 与当前已选出的所有质心之间的最短 Euclidean 距离

3. 计算每个样本被选为下一个质心的概率 $P(x_i) = \frac{D(x_i)^2}{\sum_{j} D(x_j)^2}$，采用轮盘赌选择法选出下一个质心。

* P(x_i): 样本 x_i 被选为下一个聚类质心的概率
* D(x_i): 样本 x_i 到最近已知质心的距离
* D(x_j): 数据集中样本 x_j 到最近已知质心的距离
* j: 数据集中所有样本的遍历索引

4. 重复步骤 2 和 3，直到选满 $k$ 个聚类中心。

* k: 预先设定的总聚类中心数量

5. 之后执行标准 K-Means 算法迭代更新步骤。

---

## 2. 数学公式与推导

1. **最短距离计算**：

$$D(x_i) = \min_{j \in \{1, \dots, m\}} \Vert{}x_i - \mu_j\Vert{}_2$$

* D(x_i): 样本 x_i 到当前已选质心集合的最短距离
* x_i: 第 i 个数据样本
* μ(缪)_j: 当前已选出的第 j 个聚类质心
* m: 当前已经选出的质心总数
* j: 已选质心的索引集合 $\{1, \dots, m\}$ 中的索引
* |\cdot|_2: L2 范数（即欧氏距离）

2. **抽样概率计算**：

$$P(x_i) = \frac{D(x_i)^2}{\sum_{k=1}^N D(x_k)^2}$$

* P(x_i): 样本 x_i 被选为下一个聚类质心的概率
* D(x_i): 样本 x_i 到最近质心的距离
* D(x_k): 样本 x_k 到最近质心的距离
* k: 数据集中所有样本的遍历索引 ($1$ 到 $N$)
* N: 数据集中样本的总数量

---

## 3. ASCII 流程图

```
 [随机选取第1个质心 μ1]
         |
         v
 [计算每个样本到已有质心的最近距离 D(x)]
         |
         v
 [根据概率 P(x) ∝ D(x)^2 采样下一个质心]
         |
         +<-- [未达到 K 个质心?]
         |          | 是
         |          +------+
         | 否
         v
 [运行常规 K-Means 迭代直到收敛]


```

---

## 4. Python 代码实现 (基于 NumPy / Scikit-Learn)

### 4.1 NumPy 实现 K-Means++ 初始化

```python
import numpy as np

def kmeans_plusplus_init(X, k):
    n_samples, n_features = X.shape
    centers = np.empty((k, n_features))
    
    random_idx = np.random.randint(n_samples)
    centers[0] = X[random_idx]
    
    for c_idx in range(1, k):
        dist_sq = np.array([min([np.sum((x - center)**2) for center in centers[:c_idx]]) for x in X])
        probs = dist_sq / np.sum(dist_sq)
        cum_probs = np.cumsum(probs)
        r = np.random.rand()
        
        next_idx = np.searchsorted(cum_probs, r)
        centers[c_idx] = X[next_idx]
        
    return centers

if __name__ == "__main__":
    np.random.seed(42)
    X = np.vstack([np.random.randn(50, 2) + [5, 5],
                   np.random.randn(50, 2) + [-5, -5]])
    
    initial_centers = kmeans_plusplus_init(X, k=2)
    print("K-Means++ 初始质心:\n", initial_centers)


```