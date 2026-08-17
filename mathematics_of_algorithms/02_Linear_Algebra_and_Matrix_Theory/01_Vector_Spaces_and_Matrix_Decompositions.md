# 第一章：向量空间、特征分解与三大基本矩阵分解 (SVD, QR, LU)

## 1. 核心概念与数学表达

### 1.1 向量空间与四大基本子空间
**向量空间**（Vector Space，也称线性空间）是线性代数中最核心的抽象概念之一。

简单来说，**向量空间是一个集合，其中的对象（称为“向量”）可以任意进行“加法”和“数乘（乘以标量）”两种运算，且运算结果依然留在该集合内部，同时满足特定的 8 条代数规则。**

---

### 1. 核心定义：封闭性与 8 条公理

假设有一个集合 $V$ 以及一个数域 $F$（通常是实数域 $\mathbb{R}$ 或复数域 $\mathbb{C}$）。若 $V$ 是向量空间，首先必须满足**运算封闭性**：

* **加法封闭**：对任意 $u, v \in V$，其和 $u + v \in V$。
* **数乘封闭**：对任意 $c \in F$ 和 $v \in V$，其积 $c v \in V$。

同时，这两种运算必须严格遵循以下 **8 条公理**：

**加法运算律**

1. **交换律**：$u + v = v + u$
2. **结合律**：$(u + v) + w = u + (v + w)$
3. **零元存在性**：存在一个零向量 $\mathbf{0} \in V$，使得对任意 $v \in V$，都有 $v + \mathbf{0} = v$
4. **负元存在性**：对任意 $v \in V$，都存在负向量 $-v \in V$，使得 $v + (-v) = \mathbf{0}$

**数乘运算律**

5. **数乘结合律**：$a(bv) = (ab)v$
6. **标量分配律**：$(a + b)v = av + bv$
7. **向量分配律**：$a(u + v) = au + av$
8. **单位标量乘法**：$1 \cdot v = v$（其中 $1$ 是数域 $F$ 中的乘法单位元）

向量空间（Vector Space）是一个三元组 $(V, F, +_{v}, \cdot_{s})$：$V$ 是一个集合（元素称为向量）；$F$ 是一个数域（如实数域 $\mathbb{R}$）；$+_{v}$ 和 $\cdot_{s}$ 是定义在这个集合上的加法与数乘运算。单个矩阵本身不是向量空间，但“所有同尺寸矩阵组成的集合”构成了向量空间；同时，某些具有特定约束条件的矩阵集合（如对称矩阵集合）构成了矩阵空间的“子空间”。

---

### 2. 直观理解：从“箭头”到“抽象对象”

人们通常最早接触的向量是几何中的“带箭头的线段”（如二维平面 $\mathbb{R}^2$ 或三维空间 $\mathbb{R}^3$），但向量空间的真正强大之处在于它的**抽象性**——只要满足上述公理，集合里的元素不一定非得是“箭头”或“有序数组”：

| 集合类型 | 向量的具体形态 | 加法与数乘运算 |
| --- | --- | --- |
| **坐标空间 $\mathbb{R}^n$** | $n$ 维数组，如 $(x_1, x_2, \dots, x_n)$ | 对应分量相加/相乘 |
| **多项式空间 $\mathcal{P}_n$** | 最高次数不超过 $n$ 的多项式，如 $p(x) = 2x^2 + 3x - 1$ | 同类项合并，系数相乘 |
| **矩阵空间 $\mathbb{R}^{m \times n}$** | $m \times n$ 的实数矩阵 | 矩阵对应元素相加，标量乘矩阵 |
| **函数空间 $C[a, b]$** | 在区间 $[a, b]$ 上连续的函数，如 $f(x) = \sin(x)$ | 函数值的逐点相加与数乘 |

---

### 3. 向量空间的关键衍生概念

在向量空间中，有几个非常基础且核心的概念用来描述其内部结构：

* **子空间 (Subspace)**：向量空间 $V$ 的一个非空子集 $W$，如果对 $V$ 中的加法和数乘运算也保持封闭，那么 $W$ 本身也是一个向量空间（例如：三维空间中穿过原点的一条直线或一个平面）。
* **线性组合 (Linear Combination)**：形如 $c_1 v_1 + c_2 v_2 + \dots + c_k v_k$ 的表达式。
* **张成空间 (Span)**：一组向量的所有可能线性组合构成的集合，它必然是一个子空间。
* **线性无关 (Linear Independence)**：如果 $c_1 v_1 + c_2 v_2 + \dots + c_k v_k = \mathbf{0}$ 当且仅当 $c_1 = c_2 = \dots = c_k = 0$ 时成立，则称这组向量线性无关（即没有冗余向量）。
* **基 (Basis) 与维数 (Dimension)**：
* **基**是既能张成整个向量空间、又线性无关的一组向量集合。
* **维数**即基中所含向量的个数（如 $\mathbb{R}^3$ 的维数为 3）。

对于矩阵 $A \in \mathbb{R}^{m \times n}$，定义其引发的四大基本子空间：

1. **列空间 (Column Space)**：$\mathcal{C}(A) = \{Ax \mid x \in \mathbb{R}^n\} \subseteq \mathbb{R}^m$，维数为 $\text{rank}(A)$。
2. **零空间 (Nullspace)**：$\mathcal{N}(A) = \{x \in \mathbb{R}^n \mid Ax = 0\} \subseteq \mathbb{R}^n$，维数为 $n - \text{rank}(A)$。
3. **行空间 (Row Space)**：$\mathcal{C}(A^T) = \{A^T y \mid y \in \mathbb{R}^m\} \subseteq \mathbb{R}^n$，维数为 $\text{rank}(A)$。
4. **左零空间 (Left Nullspace)**：$\mathcal{N}(A^T) = \{y \in \mathbb{R}^m \mid A^T y = 0\} \subseteq \mathbb{R}^m$，维数为 $m - \text{rank}(A)$。

**正交直和定理**：


$$\mathcal{C}(A^T) \perp \mathcal{N}(A), \quad \mathbb{R}^n = \mathcal{C}(A^T) \oplus \mathcal{N}(A)$$

$$\mathcal{C}(A) \perp \mathcal{N}(A^T), \quad \mathbb{R}^m = \mathcal{C}(A) \oplus \mathcal{N}(A^T)$$

### 1.2 特征分解 (Eigendecomposition)

若 $A \in \mathbb{R}^{n \times n}$ 有 $n$ 个线性无关的特征向量，则 $A$ 可被对角化：


$$A = V \Lambda V^{-1}$$


其中 $V = [v_1, v_2, \dots, v_n]$ 是特征向量组成的矩阵，$\Lambda = \text{diag}(\lambda_1, \dots, \lambda_n)$ 为特征值对角阵。

* **对称矩阵谱定理 (Spectral Theorem)**：若 $A = A^T$，则存在正交矩阵 $Q$ ($Q^T Q = I$) 使得 $A = Q \Lambda Q^T$。

---

## 2. 三大核心矩阵分解推导

### 2.1 LU 分解与 PLU 分解

将方阵 $A$ 分解为一个下三角矩阵 $L$ 和上三角矩阵 $U$：


$$A = L U \quad (\text{带主元选择时: } P A = L U)$$

### 2.2 QR 分解 (Gram-Schmidt 正交化)

将 $A \in \mathbb{R}^{m \times n}$ ($m \ge n$) 分解为正交矩阵 $Q \in \mathbb{R}^{m \times n}$ ($Q^T Q = I_n$) 与上三角矩阵 $R \in \mathbb{R}^{n \times n}$：


$$A = Q R$$

### 2.3 奇异值分解 (Singular Value Decomposition, SVD)

对任意矩阵 $A \in \mathbb{R}^{m \times n}$，存在正交矩阵 $U \in \mathbb{R}^{m \times m}$ 和 $V \in \mathbb{R}^{n \times n}$，使得：


$$A = U \Sigma V^T$$


其中 $\Sigma \in \mathbb{R}^{m \times n}$ 对角线上为奇异值 $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ ($r = \text{rank}(A)$)。

* **数学关系**：
* $A A^T = U \Sigma \Sigma^T U^T \implies U$ 的列是 $A A^T$ 的特征向量（左奇异向量）。
* $A^T A = V \Sigma^T \Sigma V^T \implies V$ 的列是 $A^T A$ 的特征向量（右奇异向量）。
* 奇异值 $\sigma_i = \sqrt{\lambda_i(A^T A)}$。



---

## 3. AI/ML 经典应用案例：主成分分析 (PCA) 与 低秩近似压缩 (Eckart-Young 定理)

### 3.1 Eckart-Young-Mirsky 定理

设 $A = \sum_{i=1}^r \sigma_i u_i v_i^T$ 为 $A$ 的 SVD。定义 $A_k = \sum_{i=1}^k \sigma_i u_i v_i^T$ ($k < r$) 为截断 SVD。则 $A_k$ 是秩为 $k$ 的最佳近似矩阵：


$$\min_{\text{rank}(B) \le k} \|A - B\|_F = \|A - A_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}$$

### 3.2 矩阵分解与 PCA 从零实现 Python 代码

```python
import numpy as np

class PrincipalComponentAnalysisSVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.singular_values = None

    def fit(self, X):
        # 1. 中心化数据
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        N = X.shape[0]
        
        # 2. 对中心化矩阵进行 SVD: X_centered = U * S * V^T
        # numpy.linalg.svd 返回的 Vh 是 V^T
        U, S, Vh = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. 主成分即为右奇异向量 V (Vh 的行)
        self.components = Vh[:self.n_components]
        # 方差解释: lambda_i = S_i^2 / (N - 1)
        self.explained_variance = (S[:self.n_components] ** 2) / (N - 1)
        self.singular_values = S[:self.n_components]

    def transform(self, X):
        X_centered = X - self.mean
        # 投影到主成分空间: X_new = X_centered * V_k
        return np.dot(X_centered, self.components.T)

    def inverse_transform(self, X_transformed):
        # 重构原始数据: X_reconstructed = X_transformed * V_k^T + mean
        return np.dot(X_transformed, self.components) + self.mean

# 验证 PCA 低秩重构
np.random.seed(42)
X_dummy = np.random.randn(100, 20) # 100个样本，20维特征

pca = PrincipalComponentAnalysisSVD(n_components=5)
pca.fit(X_dummy)
X_reduced = pca.transform(X_dummy)
X_reconstructed = pca.inverse_transform(X_reduced)

reconstruction_error = np.linalg.norm(X_dummy - X_reconstructed, 'fro')
print(f"原始形状: {X_dummy.shape} -> 降维后形状: {X_reduced.shape}")
print(f"Frobenius 范数重构误差 (前5主成分): {reconstruction_error:.4f}")

```

