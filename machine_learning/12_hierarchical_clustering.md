# 12. 层次聚类 (Hierarchical Clustering)

## 1. 核心原理

层次聚类通过构建聚类树（Dendrogram）来对数据集进行层级划分。主要分为两大类：

1. **凝聚（Agglomerative）**：自底向上（Bottom-Up）。开始时每个点自成一簇，然后反复合并距离最近的两个簇，直到只剩下一个簇或达到设定条件。
2. **分裂（Divisive）**：自顶向下（Top-Down）。开始时所有点在一个簇中，然后逐步递归拆分为更小的簇。

常用的簇间距离链接方式（Linkage Criteria）：

* **Single Linkage**：两簇中最近点之间的距离。
* **Complete Linkage**：两簇中最远点之间的距离。
* **Average Linkage**：两簇中所有点对距离的平均值。
* **Ward's Linkage**：合并后簇内方差增加量最小化。

## 2. 算法与数学公式

### 2.1 欧氏距离

$$d(\mathbf{x}, \mathbf{y}) = \Vert{}\mathbf{x} - \mathbf{y}\Vert{}_2$$

* $d(\mathbf{x}, \mathbf{y})$: 样本点 $\mathbf{x}$ 与样本点 $\mathbf{y}$ 之间的欧氏距离
* $\mathbf{x}$: 第一个数据点的特征向量
* $\mathbf{y}$: 第二个数据点的特征向量
* $\Vert{}\cdot{}\Vert{}_2$: L2 范数（欧氏长度）

### 2.2 簇间距离定义 (Linkage Metrics)

* **单链接 (Single)**:

$$D(A, B) = \min_{\mathbf{x} \in A, \mathbf{y} \in B} d(\mathbf{x}, \mathbf{y})$$

* $D(A, B)$: 簇 $A$ 与簇 $B$ 之间的单链接距离
* $A$: 第一个样本簇
* $B$: 第二个样本簇
* $\mathbf{x}$: 属于簇 $A$ 的数据点向量
* $\mathbf{y}$: 属于簇 $B$ 的数据点向量
* $d(\mathbf{x}, \mathbf{y})$: 数据点 $\mathbf{x}$ 与数据点 $\mathbf{y}$ 之间的距离
* $\min$: 最小值函数，提取所有点对中的最近距离

* **全链接 (Complete)**:

$$D(A, B) = \max_{\mathbf{x} \in A, \mathbf{y} \in B} d(\mathbf{x}, \mathbf{y})$$

* $D(A, B)$: 簇 $A$ 与簇 $B$ 之间的全链接距离
* $A$: 第一个样本簇
* $B$: 第二个样本簇
* $\mathbf{x}$: 属于簇 $A$ 的数据点向量
* $\mathbf{y}$: 属于簇 $B$ 的数据点向量
* $d(\mathbf{x}, \mathbf{y})$: 数据点 $\mathbf{x}$ 与数据点 $\mathbf{y}$ 之间的距离
* $\max$: 最大值函数，提取所有点对中的最远距离

* **均值链接 (Average)**:

$$D(A, B) = \frac{1}{\vert{}A\vert{}\vert{}B\vert{}} \sum_{\mathbf{x} \in A} \sum_{\mathbf{y} \in B} d(\mathbf{x}, \mathbf{y})$$

* $D(A, B)$: 簇 $A$ 与簇 $B$ 之间的均值链接距离
* $\vert{}A\vert{}$: 簇 $A$ 中的样本数量（簇大小）
* $\vert{}B\vert{}$: 簇 $B$ 中的样本数量（簇大小）
* $\mathbf{x}$: 属于簇 $A$ 的数据点向量
* $\mathbf{y}$: 属于簇 $B$ 的数据点向量
* $d(\mathbf{x}, \mathbf{y})$: 数据点 $\mathbf{x}$ 与数据点 $\mathbf{y}$ 之间的距离
* $\sum$: 求和符号，累加簇 $A$ 与簇 $B$ 中所有数据点对的距离

* **Ward 最小方差法**:

$$\Delta \text{ESS}_{AB} = \frac{\vert{}A\vert{}\vert{}B\vert{}}{\vert{}A\vert{} + \vert{}B\vert{}} \Vert{}\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\Vert{}^2$$

* $\Delta \text{ESS}_{AB}$: 合并簇 $A$ 和簇 $B$ 所导致的误差平方和（Error Sum of Squares）增加量
* $\vert{}A\vert{}$: 簇 $A$ 中的样本数量
* $\vert{}B\vert{}$: 簇 $B$ 中的样本数量
* $\boldsymbol{\mu}_A$(缪): 簇 $A$ 中所有数据点的均值向量（中心点）
* $\boldsymbol{\mu}_B$(缪): 簇 $B$ 中所有数据点的均值向量（中心点）
* $\Vert{}\boldsymbol{\mu}_A - \boldsymbol{\mu}_B\Vert{}^2$: 两簇中心点之间的欧氏距离平方

## 3. ASCII 树状图/流程框架图

```
层次聚类树状图 (Dendrogram):
              +---------------+
              |   Cluster ABC |  (最终合并)
              +-------+-------+
                      |
           +----------+----------+
           |                     |
     +-----+-----+         +-----+-----+
     | Cluster AB|         | Point C   |
     +-----+-----+         +-----------+
           |
     +-----+-----+
     |           |
 +---+---+   +---+---+
 | Point A|  | Point B|
 +-------+   +-------+

凝聚聚类流程:
 [每个点为一个簇] --> [计算所有簇间距离] --> [合并最近的两个簇] --> [更新距离矩阵] --> [重复直到K个簇]


```

## 4. Scikit-Learn 与 SciPy 代码实现

```python
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# 生成示例数据
np.random.seed(42)
X = np.random.rand(10, 2)

# 使用 sklearn 凝聚层次聚类
agg_clustering = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = agg_clustering.fit_predict(X)

print("Cluster labels assigned by sklearn AgglomerativeClustering:")
print(labels)

# 使用 scipy 生成层次结构矩阵（用于绘制树状图）
Z = linkage(X, method='ward')
print("\nLinkage matrix Z (first 3 rows):")
print(Z[:3])


```