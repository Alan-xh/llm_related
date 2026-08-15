# 第二章：矩阵求导、微积分与向量/矩阵范数

## 1. 核心概念与数学表达

### 1.1 向量与矩阵范数 (Norms)

* **向量 $p$-范数**：$\Vert{}x\Vert{}_p = \left( \sum_{i=1}^n \vert{}x_i\vert{}^p \right)^{1/p}$
* $L_1$ 范数 (Manhattan)：$\Vert{}x\Vert{}_1 = \sum \vert{}x_i\vert{}$
* $L_2$ 范数 (Euclidean)：$\Vert{}x\Vert{}_2 = \sqrt{\sum x_i^2}$
* $L_\infty$ 范数：$\Vert{}x\Vert{}_\infty = \max_i \vert{}x_i\vert{}$


* **矩阵范数**：
* **Frobenius 范数**：$\Vert{}A\Vert{}_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2} = \sqrt{\text{tr}(A^T A)} = \sqrt{\sum_{i=1}^r \sigma_i^2}$
* **核范数 (Nuclear Norm)**：$\Vert{}A\Vert{}_* = \sum_{i=1}^r \sigma_i$（低秩矩阵优化的凸松弛）
* **谱范数 (Spectral Norm)**：$\Vert{}A\Vert{}_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$



### 1.2 矩阵求导布局 (Layout Conventions)

* **分子布局 (Numerator Layout)**：$\frac{\partial y}{\partial x}$ 的维度为 $\text{dim}(y) \times \text{dim}(x)$。
* **分母布局 (Denominator Layout)**：$\frac{\partial y}{\partial x}$ 的维度为 $\text{dim}(x) \times \text{dim}(y)$（机器学习中梯度的常用布局，使得标量对向量求导的梯度方向与向量自身维度一致）。

---

## 2. 经典矩阵求导公式推导（分母布局）

设 $x, y \in \mathbb{R}^n$，$A, B \in \mathbb{R}^{n \times n}$，$W \in \mathbb{R}^{m \times n}$：

1. **线性映射梯度**：

$$\frac{\partial (a^T x)}{\partial x} = a, \quad \frac{\partial (x^T A y)}{\partial x} = A y$$


2. **二次型梯度**：

$$\frac{\partial (x^T A x)}{\partial x} = (A + A^T) x$$



若 $A$ 是对称阵（$A = A^T$），则 $\frac{\partial (x^T A x)}{\partial x} = 2 A x$。
3. **矩阵迹的求导法则 (Trace Operations)**：
* $\frac{\partial \text{tr}(W A)}{\partial W} = A^T$
* $\frac{\partial \text{tr}(W^T A W)}{\partial W} = A W + A^T W$
* $\frac{\partial \text{tr}(A W B W^T)}{\partial W} = A^T W B^T + A W B$


4. **矩阵范数平方导数**：

$$\frac{\partial \|X A - B\|_F^2}{\partial X} = 2 (X A - B) A^T$$



---

## 3. AI/ML 经典应用案例：多层感知机 (MLP) 全矩阵化反向传播算法推导与 NumPy 实现

### 3.1 前向传播与损失定义

假设网络层 $l$：


$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$

$$A^{[l]} = \sigma(Z^{[l]})$$


其中批次大小为 $B$，输入维度 $d_{l-1}$，输出维度 $d_l$。矩阵维度：
$A^{[l-1]} \in \mathbb{R}^{B \times d_{l-1}}$，$W^{[l]} \in \mathbb{R}^{d_{l-1} \times d_l}$，$b^{[l]} \in \mathbb{R}^{1 \times d_l}$。

损失函数为 $J(W, b)$。定义敏感度矩阵（Error Tensor）：


$$\delta^{[l]} = \frac{\partial J}{\partial Z^{[l]}} \in \mathbb{R}^{B \times d_l}$$

### 3.2 矩阵反向传播链式法则推导

1. **对权重的梯度**：

$$\frac{\partial J}{\partial W^{[l]}} = (A^{[l-1]})^T \delta^{[l]} \quad \in \mathbb{R}^{d_{l-1} \times d_l}$$


2. **对偏置的梯度**：

$$\frac{\partial J}{\partial b^{[l]}} = \sum_{i=1}^B \delta_i^{[l]} = \mathbf{1}_{1 \times B} \delta^{[l]} \quad \in \mathbb{R}^{1 \times d_l}$$


3. **敏感度向前一层的回传**：

$$\delta^{[l-1]} = \frac{\partial J}{\partial Z^{[l-1]}} = \left( \delta^{[l]} (W^{[l]})^T \right) \odot \sigma'(Z^{[l-1]})$$



其中 $\odot$ 表示 Hadamard 积（逐元素相乘）。

### 3.3 纯向量化反向传播完整代码实现

```python
import numpy as np

class ReluDenseLayer:
    def __init__(self, in_features, out_features):
        # He 正态分布初始化
        self.W = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.b = np.zeros((1, out_features))
        self.dW = None
        self.db = None
        
    def forward(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.dot(A_prev, self.W) + self.b
        self.A = np.maximum(0, self.Z) # ReLU 激活
        return self.A

    def backward(self, dA):
        # ReLU 的导数
        dZ = dA * (self.Z > 0).astype(np.float64)
        
        # 矩阵求导计算梯度
        B = self.A_prev.shape[0]
        self.dW = np.dot(self.A_prev.T, dZ) / B
        self.db = np.sum(dZ, axis=0, keepdims=True) / B
        
        # 梯度回传给上一层
        dA_prev = np.dot(dZ, self.W.T)
        return dA_prev

# 测试反向传播梯度流
np.random.seed(42)
B_size, D_in, D_hidden, D_out = 32, 128, 64, 10

X_batch = np.random.randn(B_size, D_in)
y_true = np.random.randn(B_size, D_out)

layer1 = ReluDenseLayer(D_in, D_hidden)
layer2 = ReluDenseLayer(D_hidden, D_out)

# 前向计算
h1 = layer1.forward(X_batch)
out = layer2.forward(h1)

# 计算 MSE 损失及其衍生梯度
loss = np.mean((out - y_true) ** 2)
dOut = 2.0 * (out - y_true) / D_out

# 梯度的全矩阵反向传递
dh1 = layer2.backward(dOut)
dX = layer1.backward(dh1)

print(f"前向损失: {loss:.6f}")
print(f"Layer1 dW 形状: {layer1.dW.shape} (期望: {D_in}, {D_hidden})")
print(f"Layer2 dW 形状: {layer2.dW.shape} (期望: {D_hidden}, {D_out})")

```

