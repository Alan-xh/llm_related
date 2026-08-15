
# 14. 主成分分析 (Principal Component Analysis, PCA)

## 1. 核心原理

PCA 是一种无监督的线性降维算法。其核心原理是将高维特征空间投影到低维正交子空间上，同时最大化投影后数据的方差（最大方差理论），或者等价地最小化数据重建误差（最小均方误差）。

主要步骤：

1. 数据中心化（去均值）。
2. 计算特征协方差矩阵。
3. 对协方差矩阵进行特征值分解（或对数据矩阵进行 SVD 奇异值分解）。
4. 选取特征值最大的前 $k$ 个特征向量作为主成分方向。
5. 将原始数据投影到由这 $k$ 个特征向量构建的正交子空间中。

## 2. 算法与数学公式

### 2.1 数据中心化

$$\mathbf{X}_{\text{centered}} = \mathbf{X} - \bar{\mathbf{X}}$$

### 2.2 协方差矩阵

$$\mathbf{\Sigma} = \frac{1}{n-1} \mathbf{X}_{\text{centered}}^T \mathbf{X}_{\text{centered}}$$

### 2.3 特征分解

$$\mathbf{\Sigma} \mathbf{v}_i = \lambda_i \mathbf{v}_i$$


按特征值大小排序：$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_D \ge 0$。

### 2.4 方差解释率 (Explained Variance Ratio)

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{j=1}^{D} \lambda_j}$$

### 2.5 投影降维

选用前 $k$ 个最大特征值对应的特征向量构成矩阵 $\mathbf{W}_k \in \mathbb{R}^{D \times k}$：


$$\mathbf{Z} = \mathbf{X}_{\text{centered}} \mathbf{W}_k$$

## 3. ASCII 流程框架图

```
+-----------------------------+
|  数据矩阵 X (n_samples, D)   |
+--------------+--------------+
               |
               v
+-----------------------------+
|    数据中心化 (减去均值)     |
+--------------+--------------+
               |
               v
+-----------------------------+
|   计算协方差矩阵 Sigma      |
+--------------+--------------+
               |
               v
+-----------------------------+
| 特征值分解 / SVD 奇异值分解  |
+--------------+--------------+
               |
               v
+-----------------------------+
| 选取前 k 个最大的特征向量 W  |
+--------------+--------------+
               |
               v
+-----------------------------+
| 降维投影 Z = X_centered * W |
+-----------------------------+

```

## 4. NumPy 纯代码实现

```python
import numpy as np

class PCAFromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ratio = None

    def fit(self, X):
        # 1. 中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. 计算协方差矩阵
        cov_matrix = np.cov(X_centered, rowvar=False)

        # 3. 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 4. 从大到小排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. 截取前 n_components 个特征向量
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.multivariate_normal(mean=[0, 0, 0], cov=[[3,1,0],[1,2,0],[0,0,1]], size=100)
    
    pca = PCAFromScratch(n_components=2)
    pca.fit(X)
    X_reduced = pca.transform(X)
    
    print("Reduced Shape:", X_reduced.shape)
    print("Explained Variance Ratios:", pca.explained_variance_ratio)

```

