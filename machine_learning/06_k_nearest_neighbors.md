# K最近邻 (K-Nearest Neighbors, KNN)

## 1. 算法原理与概述

KNN 是一种基于实例（Instance-based）的惰性学习（Lazy Learning）算法。它没有显式的训练阶段，对于未知输入，利用特征空间中与该样本距离最近的 $K$ 个已知训练样本，通过多数投票或距离加权的方式进行分类或回归。

* K: 选取的最近邻样本的数量

```
                        (未知测试样本 ?)
                            /   |   \
                           /    |    \ (计算欧氏距离)
                          v     v     v
                     +-------+-------+-------+
                     | 近邻1 | 近邻2 | 近邻3 |  (K=3)
                     +-------+-------+-------+
                          |       |       |
                          v       v       v
                        类别 A  类别 A  类别 B
                          \       |       /
                           v      v      v
                     [ 多数表决: 归为类别 A ]


```

---

## 2. 数学原理

### 2.1 距离度量 (Minkowski Distance)

$$D(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{n} \vert{}x_i - y_i\vert{}^p \right)^{\frac{1}{p}}$$

* D(x, y): 样本向量 $\mathbf{x}$ 与 $\mathbf{y}$ 之间的闵可夫斯基距离
* x: 样本向量 $\mathbf{x}$
* y: 样本向量 $\mathbf{y}$
* i: 特征维度的索引
* n: 特征的总维度/特征数量
* x_i: 样本 $\mathbf{x}$ 在第 $i$ 个维度上的特征值
* y_i: 样本 $\mathbf{y}$ 在第 $i$ 个维度上的特征值
* p: 距离度量的阶数参数

* $p=1$：曼哈顿距离（Manhattan Distance）

* p: 距离度量阶数，取值为 1

* $p=2$：欧氏距离（Euclidean Distance）

* p: 距离度量阶数，取值为 2

### 2.2 决策规则

分类预测函数：

$$y = \arg\max_c \sum_{\mathbf{x}_i \in N_K(\mathbf{x})} I(y_i = c)$$

* y: 未知测试样本 $\mathbf{x}$ 的预测类别
* argmax_c: 使得后方表达式达到最大值时对应的类别 $c$
* c: 候选的类别标签
* x_i: 属于测试样本 $\mathbf{x}$ 的 $K$ 个最近邻集合中的第 $i$ 个训练样本
* N_K(x): 测试样本 $\mathbf{x}$ 的 $K$ 个最近邻样本构成的集合
* I(·): 指示函数（Indicator Function），当条件成立时取值为 1，否则为 0
* y_i: 训练样本 $\mathbf{x}_i$ 的真实类别标签

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.metrics import accuracy_score

class CustomKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # 计算与训练数据的欧氏距离
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # 获取前 k 个最小距离的索引
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    custom_knn = CustomKNN(k=5)
    custom_knn.fit(X, y)
    print("Custom KNN Acc:", accuracy_score(y, custom_knn.predict(X)))

    sk_knn = SklearnKNN(n_neighbors=5)
    sk_knn.fit(X, y)
    print("Sklearn KNN Acc:", accuracy_score(y, sk_knn.predict(X)))


```