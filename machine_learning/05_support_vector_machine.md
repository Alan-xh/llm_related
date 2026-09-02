# 支持向量机 (Support Vector Machine, SVM)

## 1. 算法原理与概述

支持向量机（SVM）通过寻找一个最大间隔超平面（Maximum Margin Hyperplane）在高维空间中分离不同类别的样本。其核心原理是最大化正负样本到决策边界的最短距离（间隔）。

```
            x  (正类)
           /  x    
          / x    /  <-- 支撑平面 w^T*x + b = 1
 ------/-----/------------- 超平面: w^T*x + b = 0 (最大间隔 margin = 2/||w||)
        /    o/    
       /   o /     <-- 支撑平面 w^T*x + b = -1
          o   (负类)


```

---

## 2. 数学原理与对偶问题

### 2.1 原始优化问题 (硬间隔)

$$\min_{\mathbf{w}, b} \frac{1}{2} \Vert{}\mathbf{w}\Vert{}^2 \quad \text{s.t.} \quad y_i (\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \; \forall i$$

* $\mathbf{w}$: 超平面的法向量，决定超平面的方向
* $b$: 超平面的偏置项（位移项），决定超平面与原点的距离
* $\Vert\mathbf{w}\Vert$: 法向量 $\mathbf{w}$ 的 $L_2$ 范数（模长）
* $y_i$: 第 $i$ 个样本的真实类别标签（通常取值为 $+1$ 或 $-1$）
* $\mathbf{x}_i$: 第 $i$ 个样本的特征向量
* $\mathbf{w}^T \mathbf{x}_i + b$: 决策函数/超平面方程
* $i$: 样本索引
* $\text{s.t.}$: subject to 的缩写，表示约束条件
* $\forall$: 全称量词，表示“对于任意”

### 2.2 凸二次规划拉格朗日函数

引入拉格朗日乘子 $\alpha_i \ge 0$：

$$L(\mathbf{w}, b, \mathbf{\alpha}) = \frac{1}{2} \Vert{}\mathbf{w}\Vert{}^2 - \sum_{i=1}^{m} \alpha_i \left[ y_i (\mathbf{w}^T \mathbf{x}_i + b) - 1 \right]$$

* $L(\mathbf{w}, b, \mathbf{\alpha})$: 拉格朗日函数
* $\mathbf{w}$: 超平面的法向量
* $b$: 超平面的偏置项
* $\mathbf{\alpha}$: 拉格朗日乘子组成的向量
* $\alpha_i(\text{阿尔法}_i)$: 对应第 $i$ 个样本约束条件的拉格朗日乘子 ($\alpha_i \ge 0$)
* $\Vert\mathbf{w}\Vert$: 法向量 $\mathbf{w}$ 的 $L_2$ 范数
* $m$: 样本总数
* $y_i$: 第 $i$ 个样本的类别标签
* $\mathbf{x}_i$: 第 $i$ 个样本的特征向量

### 2.3 对偶问题 (Dual Problem)

$$\max_{\mathbf{\alpha}} \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m}\sum_{j=1}^{m} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^T \mathbf{x}_j \quad \text{s.t.} \quad \sum_{i=1}^{m} \alpha_i y_i = 0, \; \alpha_i \ge 0$$

* $\mathbf{\alpha}$: 对偶变量（拉格朗日乘子向量）
* $\alpha_i(\text{阿尔法}_i)$: 第 $i$ 个样本对应的拉格朗日乘子
* $\alpha_j(\text{阿尔法}_j)$: 第 $j$ 个样本对应的拉格朗日乘子
* $m$: 样本总数
* $y_i$: 第 $i$ 个样本的类别标签
* $y_j$: 第 $j$ 个样本的类别标签
* $\mathbf{x}_i$: 第 $i$ 个样本的特征向量
* $\mathbf{x}_j$: 第 $j$ 个样本的特征向量
* $\mathbf{x}_i^T \mathbf{x}_j$: 样本 $\mathbf{x}_i$ 与 $\mathbf{x}_j$ 的内积（点积）

利用核函数 $K(\mathbf{x}_i, \mathbf{x}_j)$ 替代点积 $\mathbf{x}_i^T \mathbf{x}_j$ 即可实现高维非线性映射。

* $K(\mathbf{x}_i, \mathbf{x}_j)$: 核函数，计算低维空间样本映射到高维特征空间后的内积
* $\mathbf{x}_i, \mathbf{x}_j$: 样本特征向量

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class CustomLinearSVM:
    def __init__(self, lr=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = lr
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # 转换标签为 {-1, 1}
        y_hinge = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        # SGD 求解 Hinge Loss + L2 正则化
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_hinge[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_hinge[idx]))
                    self.b -= self.lr * y_hinge[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.where(np.sign(approx) <= 0, 0, 1)

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=100, centers=2, random_state=42)

    custom_svm = CustomLinearSVM(n_iters=5000)
    custom_svm.fit(X, y)
    print("Custom SVM Accuracy:", accuracy_score(y, custom_svm.predict(X)))

    sk_svm = SVC(kernel='linear')
    sk_svm.fit(X, y)
    print("Sklearn SVM Accuracy:", accuracy_score(y, sk_svm.predict(X)))


```