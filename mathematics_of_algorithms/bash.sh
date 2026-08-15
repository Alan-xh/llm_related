#!/usr/bin/env bash

# 设置脚本在遇到错误时立即退出
set -e

# 设置根输出目录
OUTPUT_DIR="."

echo "=================================================="
echo " 开始生成人工智能数学基础课程体系文档"
echo " 目标路径: ${OUTPUT_DIR}"
echo "=================================================="

# 创建各课程目录结构
mkdir -p "${OUTPUT_DIR}/01_Optimization_Theory"
mkdir -p "${OUTPUT_DIR}/02_Linear_Algebra_and_Matrix_Theory"
mkdir -p "${OUTPUT_DIR}/03_Probability_and_Stochastic_Processes"

# ==============================================================================
# 第一部分：最优化理论 (Optimization Theory)
# ==============================================================================

cat << 'EOF' > "${OUTPUT_DIR}/01_Optimization_Theory/01_Unconstrained_Optimization.md"
# 第一章：无约束最优化理论与经典算法

## 1. 核心概念与数学表达

### 1.1 无约束优化问题形式化定义
无约束优化问题的目标是在全空间 $\mathbb{R}^n$ 上寻找目标函数的全局最小值或局部最小值：
$$\min_{x \in \mathbb{R}^n} f(x)$$
其中 $f: \mathbb{R}^n \to \mathbb{R}$ 为连续可微函数。

### 1.2 极值条件 (Optimality Conditions)
* **一阶必要条件 (First-Order Necessary Condition, FONC)**：若 $x^*$ 是局部极小值点，且 $f$ 在 $x^*$ 处一阶连续可微，则梯度为零：
  $$\nabla f(x^*) = 0$$
  满足 $\nabla f(x^*) = 0$ 的点称为**驻点 (Stationary Point)** 或**临界点 (Critical Point)**。
* **二阶必要条件 (Second-Order Necessary Condition, SONC)**：若 $x^*$ 是局部极小值点，且 $f$ 二阶连续可微，则黑塞矩阵 (Hessian Matrix) 是半正定的：
  $$\nabla^2 f(x^*) \succeq 0$$
* **二阶充分条件 (Second-Order Sufficient Condition, SOSC)**：若 $\nabla f(x^*) = 0$ 且 $\nabla^2 f(x^*) \succ 0$（严格正定），则 $x^*$ 为严格局部极小值点。

### 1.3 黑塞矩阵 (Hessian Matrix) 与鞍点 (Saddle Point)
黑塞矩阵由二阶偏导数构成：
$$H = \nabla^2 f(x) = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$
若 $\nabla f(x^*) = 0$，但 $\nabla^2 f(x^*)$ 既非正定也非负定（即拥有正负混合的特征值），则 $x^*$ 为**鞍点 (Saddle Point)**。

---

## 2. 算法原理与推导

### 2.1 梯度下降法 (Gradient Descent)
* **更新公式**：$x_{k+1} = x_k - \alpha_k \nabla f(x_k)$
* **步长选择 (Line Search)**：
  * **Armijo 准则 (充分下降条件)**：$f(x_k + \alpha_k d_k) \le f(x_k) + c_1 \alpha_k \nabla f(x_k)^T d_k$，其中 $c_1 \in (0, 1)$。
  * **Wolfe 准则**：在 Armijo 条件基础上增加曲率条件 $\nabla f(x_k + \alpha_k d_k)^T d_k \ge c_2 \nabla f(x_k)^T d_k$，其中 $c_1 < c_2 < 1$。

### 2.2 牛顿法 (Newton's Method)
利用 Taylor 展式在 $x_k$ 处二阶展开：
$$f(x_k + d) \approx f(x_k) + \nabla f(x_k)^T d + \frac{1}{2} d^T \nabla^2 f(x_k) d$$
令导数为零求极小值，导出牛顿方向 $d_k = - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$。
* **更新公式**：$x_{k+1} = x_k - [\nabla^2 f(x_k)]^{-1} \nabla f(x_k)$
* **优点**：具有二阶收敛速度 (Quadratic Convergence)。
* **缺点**：计算并求逆 $n \times n$ 的黑塞矩阵开销巨大，复杂度为 $\mathcal{O}(n^3)$；在非凸区域，Hessian 矩阵非正定会导致搜索方向不下降。

### 2.3 拟牛顿法 (Quasi-Newton Methods)
通过构造近似矩阵 $B_k \approx \nabla^2 f(x_k)$ 或 $H_k \approx [\nabla^2 f(x_k)]^{-1}$ 避免计算二阶导数和求逆。
* **割线方程 (Secant Equation)**：设 $s_k = x_{k+1} - x_k$，$y_k = \nabla f(x_{k+1}) - \nabla f(x_k)$，要求 $B_{k+1} s_k = y_k$ 或 $H_{k+1} y_k = s_k$。
* **BFGS 算法更新公式**（更新 $H_k \approx B_k^{-1}$）：
  $$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T, \quad \rho_k = \frac{1}{y_k^T s_k}$$
* **L-BFGS (Limited-memory BFGS)**：不显式存储 $n \times n$ 的 $H_k$，仅保存最近 $m$ 步的向量对 $\{s_i, y_i\}_{i=k-m}^{k-1}$，利用双重循环递归算法计算搜索方向，将空间复杂度降至 $\mathcal{O}(mn)$。

---

## 3. AI/ML 经典应用案例：逻辑回归 (Logistic Regression) 拟牛顿法求解

### 3.1 问题建模
给定数据集 $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$，$x_i \in \mathbb{R}^d, y_i \in \{0, 1\}$。逻辑回归采用 Sigmoid 函数建模条件概率：
$$p(y_i=1\vert{}x_i; w) = \sigma(w^T x_i) = \frac{1}{1 + e^{-w^T x_i}}$$

负对数似然函数（即损失函数 $J(w)$）：
$$J(w) = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \ln \sigma(w^T x_i) + (1-y_i) \ln (1 - \sigma(w^T x_i)) \right]$$

一阶梯度向量：
$$\nabla J(w) = \frac{1}{N} \sum_{i=1}^N \left( \sigma(w^T x_i) - y_i \right) x_i = \frac{1}{N} X^T (\sigma(X w) - Y)$$

黑塞矩阵：
$$\nabla^2 J(w) = \frac{1}{N} \sum_{i=1}^N \sigma(w^T x_i)(1 - \sigma(w^T x_i)) x_i x_i^T = \frac{1}{N} X^T D X$$
其中 $D$ 是对角矩阵，对角元素 $D_{ii} = \sigma(w^T x_i)(1 - \sigma(w^T x_i)) > 0$。由于对任意非零向量 $v$，$v^T X^T D X v = (X v)^T D (X v) \ge 0$，黑塞矩阵半正定，故逻辑回归损失函数为凸函数。

### 3.2 L-BFGS 优化求解 Python 实现

```python
import numpy as np

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

def loss_and_grad(w, X, y):
    n_samples = X.shape[0]
    z = X.dot(w)
    pred = sigmoid(z)
    loss = -np.mean(y * np.log(pred + 1e-15) + (1 - y) * np.log(1 - pred + 1e-15))
    grad = (1.0 / n_samples) * X.T.dot(pred - y)
    return loss, grad

def lbfgs_two_loop(grad, s_list, y_list, m):
    q = np.copy(grad)
    alphas = []
    k = len(s_list)
    
    for i in reversed(range(k)):
        s_i = s_list[i]
        y_i = y_list[i]
        rho_i = 1.0 / np.dot(y_i, s_i)
        alpha_i = rho_i * np.dot(s_i, q)
        alphas.append(alpha_i)
        q -= alpha_i * y_i
        
    alphas.reverse()
    
    if k > 0:
        gamma = np.dot(s_list[-1], y_list[-1]) / np.dot(y_list[-1], y_list[-1])
        r = gamma * q
    else:
        r = q
        
    for i in range(k):
        s_i = s_list[i]
        y_i = y_list[i]
        rho_i = 1.0 / np.dot(y_i, s_i)
        beta = rho_i * np.dot(y_i, r)
        r += s_i * (alphas[i] - beta)
        
    return r

# 模拟数据训练
np.random.seed(42)
N, D = 1000, 10
X = np.random.randn(N, D)
true_w = np.random.randn(D)
y = (sigmoid(X.dot(true_w)) > 0.5).astype(np.float64)

# L-BFGS 优化过程
w = np.zeros(D)
m_history = 5
s_list, y_list = [], []
max_iter = 50

for iter_idx in range(max_iter):
    loss, grad = loss_and_grad(w, X, y)
    if np.linalg.norm(grad) < 1e-5:
        break
    
    r = lbfgs_two_loop(grad, s_list, y_list, m_history)
    d = -r
    
    # 线搜索 (Armijo 条件简易实现)
    alpha = 1.0
    c1 = 1e-4
    while loss_and_grad(w + alpha * d, X, y)[0] > loss + c1 * alpha * np.dot(grad, d):
        alpha *= 0.5
        
    s = alpha * d
    w_new = w + s
    _, grad_new = loss_and_grad(w_new, X, y)
    y_vec = grad_new - grad
    
    if len(s_list) >= m_history:
        s_list.pop(0)
        y_list.pop(0)
    s_list.append(s)
    y_list.append(y_vec)
    
    w = w_new
    if iter_idx % 10 == 0:
        print(f"Iteration {iter_idx:02d} | Loss: {loss:.6f}")

```

EOF

cat << 'EOF' > "${OUTPUT_DIR}/01_Optimization_Theory/02_Constrained_Optimization.md"

# 第二章：约束最优化理论与对偶性

## 1. 核心概念与数学表达

### 1.1 约束优化问题标准形式

$$\begin{aligned} \min_{x \in \mathbb{R}^n} \quad & f(x) \\ \text{s.t.} \quad & g_i(x) \le 0, \quad i = 1, \dots, m \\ & h_j(x) = 0, \quad j = 1, \dots, p \end{aligned}$$

### 1.2 拉格朗日函数 (Lagrangian)

引入拉格朗日乘子向量 $\lambda \in \mathbb{R}^m$（要求 $\lambda \ge 0$）与 $\nu \in \mathbb{R}^p$：


$$\mathcal{L}(x, \lambda, \nu) = f(x) + \sum_{i=1}^m \lambda_i g_i(x) + \sum_{j=1}^p \nu_j h_j(x)$$

### 1.3 KKT (Karush-Kuhn-Tucker) 条件

若 $x^*$ 为局部最优解，且满足正则性条件（如 Slater 条件），则存在乘子向量 $\lambda^*, \nu^*$，使得以下条件成立：

1. **平稳性条件 (Stationarity)**：

$$\nabla_x \mathcal{L}(x^*, \lambda^*, \nu^*) = \nabla f(x^*) + \sum_{i=1}^m \lambda_i^* \nabla g_i(x^*) + \sum_{j=1}^p \nu_j^* \nabla h_j(x^*) = 0$$


2. **原始可行性 (Primal Feasibility)**：

$$g_i(x^*) \le 0, \quad i = 1, \dots, m$$


$$h_j(x^*) = 0, \quad j = 1, \dots, p$$


3. **对偶可行性 (Dual Feasibility)**：

$$\lambda_i^* \ge 0, \quad i = 1, \dots, m$$


4. **互补松弛性 (Complementary Slackness)**：

$$\lambda_i^* g_i(x^*) = 0, \quad \forall i = 1, \dots, m$$



（若 $g_i(x^*) < 0$，则必有 $\lambda_i^* = 0$；若 $\lambda_i^* > 0$，则必有 $g_i(x^*) = 0$）。

---

## 2. 对偶理论 (Duality Theory)

### 2.1 拉格朗日对偶函数 (Lagrange Dual Function)

对偶函数定义为拉格朗日函数关于 $x$ 的下确界：


$$g(\lambda, \nu) = \inf_{x \in \mathcal{D}} \mathcal{L}(x, \lambda, \nu)$$


无论原问题是否为凸，对偶函数 $g(\lambda, \nu)$ 总是关于 $(\lambda, \nu)$ 的**凹函数**。

### 2.2 拉格朗日对偶问题 (Lagrange Dual Problem)

$$\max_{\lambda \succeq 0, \nu} \quad g(\lambda, \nu)$$

### 2.3 弱对偶性与强对偶性 (Weak & Strong Duality)

* **弱对偶性 (Weak Duality)**：对任意可行解 $x$ 和可行对偶变量 $(\lambda, \nu)$，恒有 $g(\lambda, \nu) \le f(x)$。故对偶问题的最优值 $d^*$ 是原问题最优值 $p^*$ 的下界：

$$d^* \le p^*$$



对偶间隙定义为 $p^* - d^* \ge 0$。
* **强对偶性 (Strong Duality)**：当 $d^* = p^*$ 时，称强对偶性成立（对偶间隙为零）。
* **Slater 条件**：对于凸优化问题，若存在严格可行点 $x \in \text{relint}(\mathcal{D})$ 使得对所有 $i$ 均有 $g_i(x) < 0$，且 $Ah_j(x) = 0$，则强对偶性成立。

---

## 3. AI/ML 经典应用案例：硬间隔与软间隔支持向量机 (SVM) 对偶推导

### 3.1 硬间隔 SVM 原问题

给定数据集 $\{(x_i, y_i)\}_{i=1}^N$，$y_i \in \{-1, +1\}$。寻找超平面 $w^T x + b = 0$ 使得分类间隔最大化：


$$\begin{aligned} \min_{w, b} \quad & \frac{1}{2} \Vert{}w\Vert{}^2 \\ \text{s.t.} \quad & 1 - y_i (w^T x_i + b) \le 0, \quad i = 1, \dots, N \end{aligned}$$

### 3.2 对偶问题详细推导

1. 构建拉格朗日函数 ($\alpha_i \ge 0$)：

$$\mathcal{L}(w, b, \alpha) = \frac{1}{2} \Vert{}w\Vert{}^2 + \sum_{i=1}^N \alpha_i \left[ 1 - y_i (w^T x_i + b) \right]$$


2. 求 $w, b$ 的偏导并令其为零：

$$\frac{\partial \mathcal{L}}{\partial w} = w - \sum_{i=1}^N \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^N \alpha_i y_i x_i$$


$$\frac{\partial \mathcal{L}}{\partial b} = -\sum_{i=1}^N \alpha_i y_i = 0 \implies \sum_{i=1}^N \alpha_i y_i = 0$$


3. 将 $w$ 代回拉格朗日函数，消去 $w$ 和 $b$：

$$\begin{aligned}    g(\alpha) &= \frac{1}{2} \left\Vert{} \sum_{i=1}^N \alpha_i y_i x_i \right\Vert{}^2 - \sum_{i=1}^N \alpha_i y_i \left( \sum_{j=1}^N \alpha_j y_j x_j \right)^T x_i - b \sum_{i=1}^N \alpha_i y_i + \sum_{i=1}^N \alpha_i \\    &= \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j x_i^T x_j    \end{aligned}$$


4. 导出对偶问题：

$$\begin{aligned}    \max_{\alpha} \quad & \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j \langle x_i, x_j \rangle \\    \text{s.t.} \quad & \sum_{i=1}^N \alpha_i y_i = 0, \quad \alpha_i \ge 0, \; i=1, \dots, N    \end{aligned}$$



引入核函数 $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ 后，即可将线性内积推广至非线性高维空间（核技巧）。

### 3.3 序列最小优化算法 (SMO) 求解核 SVM 的 Python 实现

```python
import numpy as np

class SupportVectorMachine:
    def __init__(self, C=1.0, kernel='rbf', gamma=0.1, tol=1e-3, max_passes=5):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes

    def _kernel(self, X1, X2):
        if self.kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel_type == 'rbf':
            if X1.ndim == 1:
                X1 = X1.reshape(1, -1)
            if X2.ndim == 1:
                X2 = X2.reshape(1, -1)
            sq_dists = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.X = X
        self.y = y.astype(np.float64)
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        K = self._kernel(X, X)
        passes = 0
        
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(n_samples):
                E_i = np.dot(self.alpha * self.y, K[:, i]) + self.b - self.y[i]
                
                if (self.y[i] * E_i < -self.tol and self.alpha[i] < self.C) or \
                   (self.y[i] * E_i > self.tol and self.alpha[i] > 0):
                    
                    j = np.random.choice([idx for idx in range(n_samples) if idx != i])
                    E_j = np.dot(self.alpha * self.y, K[:, j]) + self.b - self.y[j]
                    
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]
                    
                    if self.y[i] != self.y[j]:
                        L = max(0.0, self.alpha[j] - self.alpha[i])
                        H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0.0, self.alpha[i] + self.alpha[j] - self.C)
                        H = min(self.C, self.alpha[i] + self.alpha[j])
                        
                    if L == H:
                        continue
                        
                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                        
                    self.alpha[j] -= (self.y[j] * (E_i - E_j)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)
                    
                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue
                        
                    self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                    
                    b1 = self.b - E_i - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] - \
                         self.y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                        
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
                
        # 提取支持向量
        sv_indices = self.alpha > 1e-4
        self.support_vectors = X[sv_indices]
        self.sv_y = self.y[sv_indices]
        self.sv_alpha = self.alpha[sv_indices]
        print(f"训练完成，支持向量数量: {np.sum(sv_indices)} / {n_samples}")

    def predict(self, X_test):
        K_test = self._kernel(X_test, self.X)
        pred = np.dot(K_test, self.alpha * self.y) + self.b
        return np.sign(pred)

# 测试 SVM 模块
if __name__ == "__main__":
    np.random.seed(42)
    X_data = np.random.randn(100, 2)
    y_data = np.where(X_data[:, 0] + X_data[:, 1] > 0, 1.0, -1.0)
    
    svm = SupportVectorMachine(C=1.0, kernel='linear')
    svm.fit(X_data, y_data)
    preds = svm.predict(X_data)
    print(f"训练准确率: {np.mean(preds == y_data) * 100:.2f}%")

```

EOF

cat << 'EOF' > "${OUTPUT_DIR}/01_Optimization_Theory/03_Convex_Optimization_and_Proximal.md"

# 第三章：凸优化、近端算子与大规模随机优化

## 1. 核心概念与数学表达

### 1.1 凸集与凸函数

* **凸集 (Convex Set)**：集合 $C \subseteq \mathbb{R}^n$，若对任意 $x, y \in C$ 及 $\theta \in [0, 1]$，均有 $\theta x + (1-\theta)y \in C$。
* **凸函数 (Convex Function)**：定义域为凸集的函数 $f$，满足对任意 $x, y$ 及 $\theta \in [0, 1]$：

$$f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta) f(y)$$


* **一阶凸性特征**：$f(y) \ge f(x) + \nabla f(x)^T (y - x)$。
* **二阶凸性特征**：$\nabla^2 f(x) \succeq 0$（黑塞矩阵半正定）。

### 1.2 次梯度 (Subgradient) 与次微分 (Subdifferential)

对于非光滑凸函数 $f: \mathbb{R}^n \to \mathbb{R}$，向量 $g \in \mathbb{R}^n$ 称为 $f$ 在 $x$ 处的**次梯度**，若满足：


$$\forall y \in \text{dom } f, \quad f(y) \ge f(x) + g^T (y - x)$$


$f$ 在 $x$ 处的所有次梯度构成的集合称为**次微分**，记作 $\partial f(x)$。

* **极小值条件**：$x^*$ 为 $f$ 的全局最小值点当且仅当 $0 \in \partial f(x^*)$。

### 1.3 近端算子 (Proximal Operator)

对于凸函数 $h(x)$，参数 $\gamma > 0$ 下的近端算子 $\mathbf{prox}_{\gamma h}: \mathbb{R}^n \to \mathbb{R}^n$ 定义为：


$$\mathbf{prox}_{\gamma h}(v) = \arg\min_{x \in \mathbb{R}^n} \left( h(x) + \frac{1}{2\gamma} \Vert{}x - v\Vert{}_2^2 \right)$$

* **$L_1$ 正则化（Lasso）的软阈值算子 (Soft-Thresholding)**：
当 $h(x) = \lambda \Vert{}x\Vert{}_1$ 时，近端算子按元素按维解耦：

$$\left[ \mathbf{prox}_{\gamma \lambda \Vert{}\cdot\Vert{}_1}(v) \right]_i = S_{\gamma \lambda}(v_i) = \text{sign}(v_i) \max(\vert{}v_i\vert{} - \gamma \lambda, 0)$$



---

## 2. 大规模优化算法

### 2.1 随机梯度下降 (SGD) 及动量变体

目标函数：$f(w) = \frac{1}{N} \sum_{i=1}^N f_i(w)$。

* **SGD**：$w_{t+1} = w_t - \eta \nabla f_{i_t}(w_t)$
* **Polyak Momentum**：

$$v_{t+1} = \beta v_t + \eta \nabla f_{i_t}(w_t), \quad w_{t+1} = w_t - v_{t+1}$$


* **Nesterov Accelerated Gradient (NAG)**：

$$v_{t+1} = \beta v_t + \eta \nabla f_{i_t}(w_t - \beta v_t), \quad w_{t+1} = w_t - v_{t+1}$$



### 2.2 自适应学习率算法 (AdaGrad, RMSProp, Adam)

* **Adam (Adaptive Moment Estimation)**：
一阶矩估计（动量）：$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
二阶矩估计（未中心化方差）：$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
偏差修正：$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
参数更新：$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

### 2.3 复合优化与近端梯度下降 (PGD / ISTA)

针对形如 $\min_x f(x) + g(x)$ 的问题，其中 $f(x)$ 凸且可微（$L$-光滑），$g(x)$ 凸但不可微（如 $L_1$ 范数）。

* **ISTA 更新步骤**：

$$x_{k+1} = \mathbf{prox}_{\gamma g} \left( x_k - \gamma \nabla f(x_k) \right)$$



---

## 3. AI/ML 经典应用案例：Lasso 回归 (ISTA 算法) 与 自定义 PyTorch AdamW 算子

### 3.1 ISTA 算法求解 Lasso 的 Python 实现

```python
import numpy as np

def prox_l1(v, gamma_lambda):
    return np.sign(v) * np.maximum(np.abs(v) - gamma_lambda, 0.0)

def fit_lasso_ista(X, y, alpha=0.1, max_iter=200, tol=1e-5):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    
    # 计算 Lipschitz 常数 L = max_eigenvalue(X^T X / N)
    L = np.linalg.norm(X, ord=2)**2 / n_samples
    gamma = 1.0 / L
    
    for i in range(max_iter):
        w_old = w.copy()
        # f(w) 的梯度: (1/N) * X^T (X w - y)
        grad_f = (1.0 / n_samples) * X.T.dot(X.dot(w) - y)
        
        # 梯度下降 + 近端投影
        w = prox_l1(w - gamma * grad_f, gamma * alpha)
        
        if np.linalg.norm(w - w_old) < tol:
            print(f"ISTA 在第 {i} 次迭代收敛。")
            break
            
    return w

# 实验验证
np.random.seed(42)
N, D = 200, 50
X_mat = np.random.randn(N, D)
true_w = np.zeros(D)
true_w[:5] = [3.0, -2.0, 1.5, -0.8, 2.0] # 稀疏真实权重
y_vec = X_mat.dot(true_w) + 0.1 * np.random.randn(N)

estimated_w = fit_lasso_ista(X_mat, y_vec, alpha=0.05)
print(f"非零系数个数: {np.sum(estimated_w != 0)} (真实为 5)")

```

### 3.2 AdamW 优化算法底层原生 Python 完全实现

```python
class AdamWOptimizer:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        self.params = params  # 字典: {'param_name': numpy_array}
        self.grads = {}       # 字典: {'param_name': numpy_array}
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        for k in self.params.keys():
            g = grads[k]
            
            # 1. 独立解耦的权重衰减 (AdamW 的核心改动)
            self.params[k] = self.params[k] - self.lr * self.weight_decay * self.params[k]
            
            # 2. 更新一阶和二阶矩
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g ** 2)
            
            # 3. 偏差修正
            m_hat = self.m[k] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1.0 - self.beta2 ** self.t)
            
            # 4. 参数梯度更新
            self.params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# 验证 AdamW
params = {'W': np.array([2.0, -1.0]), 'b': np.array([0.5])}
opt = AdamWOptimizer(params, lr=0.1)

for step in range(5):
    # 模拟伪梯度
    dummy_grads = {'W': params['W'] * 0.5, 'b': params['b'] * 0.2}
    opt.step(dummy_grads)
    print(f"Step {opt.t} | W: {params['W']}")

```

EOF

# ==============================================================================

# 第二部分：线性代数与矩阵论 (Linear Algebra & Matrix Theory)

# ==============================================================================

cat << 'EOF' > "${OUTPUT_DIR}/02_Linear_Algebra_and_Matrix_Theory/01_Vector_Spaces_and_Matrix_Decompositions.md"

# 第一章：向量空间、特征分解与三大基本矩阵分解 (SVD, QR, LU)

## 1. 核心概念与数学表达

### 1.1 向量空间与四大基本子空间

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

EOF

cat << 'EOF' > "${OUTPUT_DIR}/02_Linear_Algebra_and_Matrix_Theory/02_Matrix_Calculus_and_Norms.md"

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

EOF

cat << 'EOF' > "${OUTPUT_DIR}/02_Linear_Algebra_and_Matrix_Theory/03_Special_Matrices_and_Low_Rank_Methods.md"

# 第三章：特殊矩阵结构、低秩分解与高维张量基础

## 1. 核心概念与特殊矩阵

### 1.1 正定与半正定矩阵 (Positive Definite Matrices)

对称矩阵 $A \in \mathbb{R}^{n \times n}$ 称为：

* **正定 ($A \succ 0$)**：若对任意非零向量 $x \in \mathbb{R}^n$，$x^T A x > 0$。 equivalent to 所有特征值 $\lambda_i > 0$。
* **半正定 ($A \succeq 0$)**：若 $x^T A x \ge 0$。 equivalent to 所有特征值 $\lambda_i \ge 0$。
* **Cholesky 分解**：若 $A \succ 0$，则存在唯一的下三角矩阵 $L$（对角线元素全正），使得 $A = L L^T$。

### 1.2 投影矩阵 (Projection Matrices) 与 伪逆 (Pseudoinverse)

* **投影矩阵 $P$**：满足自幂性 $P^2 = P$。若正交投影，则 $P = P^T$。
* 投影到子空间 $\mathcal{C}(A)$ 的矩阵：$P = A (A^T A)^{-1} A^T$。


* **Moore-Penrose 广义逆 (Pseudoinverse $A^+$)**：
对任意 $A \in \mathbb{R}^{m \times n}$，若其 SVD 为 $A = U \Sigma V^T$，则伪逆定义为：

$$A^+ = V \Sigma^+ U^T$$



其中 $\Sigma^+$ 是将 $\Sigma$ 的非零奇异值取倒数并转置得到的矩阵。
* **最小二乘通解**：对于超定方程组 $A x = b$，极小范数最小二乘解为 $\hat{x} = A^+ b$。



---

## 2. 低秩适应 (LoRA) 理论与公式证明

在大型语言模型 (LLM) 的微调过程中，假设预训练权重为 $W_0 \in \mathbb{R}^{d \times k}$。LoRA 假设权重的更新量 $\Delta W$ 具有很低的**本征秩 (Intrinsic Rank)** $r \ll \min(d, k)$。

### 2.1 参数分解形式

将 $\Delta W$ 因子分解为两个低秩矩阵的乘积：


$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$


其中：

* $A \in \mathbb{R}^{r \times k}$，通常使用高斯随机初始化 $\mathcal{N}(0, \sigma^2)$。
* $B \in \mathbb{R}^{d \times r}$，初始化为全零矩阵 $\mathbf{0}$，确保在训练开始时 $\Delta W = 0$，模型输出与原始预训练模型完全一致。
* $\alpha$ 为常数缩放系数。

### 2.2 显存与计算开销对比分析

设输入维度为 $k=4096$，输出维度 $d=4096$，秩 $r=8$：

* 原参数量：$4096 \times 4096 \approx 1.67 \times 10^7$ (16.7M)
* LoRA 参数量：$r \times (d + k) = 8 \times (4096 + 4096) = 65,536$ (0.065M)
* **参数量削减幅度**：> 99.6%！

---

## 3. AI/ML 经典应用案例：从零手写 PyTorch 风格的 LoRA 线性层

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.05):
        super(LoRALinear, self).__init__()
        
        # 1. 冻结原始预训练权重
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # 原始权重 W0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight.requires_grad = False # 冻结
        self.bias.requires_grad = False   # 冻结
        
        # 2. 可训练的低秩矩阵 A 和 B
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(p=lora_dropout)
            self.reset_lora_parameters()
            
    def reset_lora_parameters(self):
        # A 采用 Kaiming 均匀分布初始化，B 采用全零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 主路计算: x * W0^T + b
        result = nn.functional.linear(x, self.weight, self.bias)
        
        # 旁路低秩计算: (x * A^T) * B^T * scaling
        if self.rank > 0:
            lora_out = nn.functional.linear(self.dropout(x), self.lora_A) # (batch, rank)
            lora_out = nn.functional.linear(lora_out, self.lora_B)        # (batch, out_features)
            result += lora_out * self.scaling
            
        return result

# 验证 LoRA 模块
if __name__ == "__main__":
    x = torch.randn(4, 128) # Batch=4, dim=128
    lora_layer = LoRALinear(in_features=128, out_features=256, rank=8)
    
    output = lora_layer(x)
    
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
    
    print(f"输出维度: {output.shape}")
    print(f"可训练参数量 (LoRA B & A): {trainable_params}")
    print(f"冻结参数量 (W0 & bias): {frozen_params}")

```

EOF

# ==============================================================================

# 第三部分：概率论与随机过程 (Probability & Stochastic Processes)

# ==============================================================================

cat << 'EOF' > "${OUTPUT_DIR}/03_Probability_and_Stochastic_Processes/01_Probability_Distributions_and_MLE_MAP.md"

# 第一章：核心概率分布、最大似然估计与贝叶斯后验估计

## 1. 核心概念与概率分布

### 1.1 概率空间与贝叶斯定理

概率空间由元组 $(\Omega, \mathcal{F}, P)$ 定义。
**贝叶斯定理 (Bayes' Theorem)**：


$$P(\theta \mid X) = \frac{P(X \mid \theta) P(\theta)}{P(X)} = \frac{P(X \mid \theta) P(\theta)}{\int_\Theta P(X \mid \theta') P(\theta') d\theta'}$$

* $P(\theta)$：**先验概率 (Prior)**
* $P(X \mid \theta)$：**似然函数 (Likelihood)**
* $P(\theta \mid X)$：**后验概率 (Posterior)**
* $P(X)$：**边际似然/证据 (Evidence)**

### 1.2 高维多元高斯分布 (Multivariate Gaussian Distribution)

向量 $X \in \mathbb{R}^d$ 服从多元正态分布 $X \sim \mathcal{N}(\mu, \Sigma)$，其概率密度函数 (PDF) 为：


$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$


其中 $\mu \in \mathbb{R}^d$ 为均值向量，$\Sigma \in \mathbb{R}^{d \times d}$ 为协方差矩阵 ($\Sigma \succeq 0$)。指数项中的二次型 $(x - \mu)^T \Sigma^{-1} (x - \mu)$ 称为**马氏距离 (Mahalanobis Distance)** 的平方。

### 1.3 共轭先验 (Conjugate Priors)

若后验分布 $P(\theta \mid X)$ 与先验分布 $P(\theta)$ 属于同一个函数族，则称该先验与似然共轭。

* **Beta-Binomial 共轭**：似然为二项分布，先验为 Beta 分布 $\text{Beta}(\alpha, \beta)$，后验为 $\text{Beta}(\alpha + k, \beta + n - k)$。
* **Dirichlet-Multinomial 共轭**：似然为多项分布，先验为 Dirichlet 分布（Beta 在多维的推广）。

---

## 2. 估计理论：MLE, MAP 与 贝叶斯估计

### 2.1 最大似然估计 (Maximum Likelihood Estimation, MLE)

假设数据 $X = \{x_1, \dots, x_N\}$ 独立同分布 (i.i.d.)：


$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ln P(X \mid \theta) = \arg\max_\theta \sum_{i=1}^N \ln P(x_i \mid \theta)$$

### 2.2 最大后验估计 (Maximum A Posteriori, MAP)

引入参数的先验分布 $P(\theta)$：


$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \ln P(\theta \mid X) = \arg\max_\theta \left[ \sum_{i=1}^N \ln P(x_i \mid \theta) + \ln P(\theta) \right]$$

**与机器学习损失函数的正则化等价性**：

* 若先验 $P(\theta) \sim \mathcal{N}(0, \sigma_0^2 I)$（高斯先验），$\ln P(\theta) \propto -\Vert{}\theta\Vert{}_2^2 \implies$ **$L_2$ 正则化 (Ridge)**。
* 若先验 $P(\theta) \sim \text{Laplace}(0, b)$（拉普拉斯先验），$\ln P(\theta) \propto -\Vert{}\theta\Vert{}_1 \implies$ **$L_1$ 正则化 (Lasso)**。

---

## 3. AI/ML 经典应用案例：高斯混合模型 (GMM) 与 EM (期望最大化) 算法推导及 Python 实现

### 3.1 GMM 概率生成模型

高斯混合模型假设数据由 $K$ 个高斯分布的混合生成：


$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \sum_{k=1}^K \pi_k = 1$$


包含隐变量 $z_i \in \{1, \dots, K\}$ 表示样本 $x_i$ 来自第 $k$ 个高斯分量。

### 3.2 EM 算法迭代推导

* **E 步 (Expectation)**：计算隐变量的后验概率（响应度 $\gamma_{ik}$）：

$$\gamma_{ik} = P(z_i = k \mid x_i; \theta) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$


* **M 步 (Maximization)**：最大化期望对数似然，更新参数：

$$N_k = \sum_{i=1}^N \gamma_{ik}$$


$$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$


$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T$$


$$\pi_k^{\text{new}} = \frac{N_k}{N}$$



### 3.3 高斯混合模型 EM 算法完整实现

```python
import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        N, D = X.shape
        np.random.seed(42)
        
        # 参数初始化
        self.pi = np.full(self.K, 1.0 / self.K)
        random_indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[random_indices]
        self.Sigma = np.array([np.eye(D) for _ in range(self.K)])
        
        log_likelihood_old = 0
        
        for iteration in range(self.max_iter):
            # --- E 步: 计算响应度 gamma (N, K) ---
            gamma = np.zeros((N, self.K))
            for k in range(self.K):
                # 计算多元正态 PDF
                diff = X - self.mu[k]
                inv_Sigma = np.linalg.inv(self.Sigma[k] + 1e-6 * np.eye(D))
                det_Sigma = np.linalg.det(self.Sigma[k] + 1e-6 * np.eye(D))
                
                norm_const = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det_Sigma) + 1e-15)
                exon = -0.5 * np.sum(np.dot(diff, inv_Sigma) * diff, axis=1)
                gamma[:, k] = self.pi[k] * norm_const * np.exp(exon)
                
            sum_gamma = np.sum(gamma, axis=1, keepdims=True) + 1e-15
            gamma /= sum_gamma
            
            # --- M 步: 更新参数 ---
            N_k = np.sum(gamma, axis=0)
            
            for k in range(self.K):
                self.mu[k] = np.sum(gamma[:, k:], axis=0) / N_k[k] if False else np.sum(gamma[:, [k]] * X, axis=0) / N_k[k]
                diff = X - self.mu[k]
                self.Sigma[k] = np.dot((gamma[:, [k]] * diff).T, diff) / N_k[k]
                self.pi[k] = N_k[k] / N
                
            # 计算对数似然
            log_likelihood = np.sum(np.log(sum_gamma))
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"GMM 在第 {iteration} 次迭代收敛。")
                break
            log_likelihood_old = log_likelihood

    def predict_proba(self, X):
        N, D = X.shape
        gamma = np.zeros((N, self.K))
        for k in range(self.K):
            diff = X - self.mu[k]
            inv_Sigma = np.linalg.inv(self.Sigma[k] + 1e-6 * np.eye(D))
            det_Sigma = np.linalg.det(self.Sigma[k] + 1e-6 * np.eye(D))
            norm_const = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det_Sigma) + 1e-15)
            exon = -0.5 * np.sum(np.dot(diff, inv_Sigma) * diff, axis=1)
            gamma[:, k] = self.pi[k] * norm_const * np.exp(exon)
        return gamma / np.sum(gamma, axis=1, keepdims=True)

# 验证 GMM 聚类
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([3, 3])
X2 = np.random.randn(100, 2) + np.array([-3, -3])
X_data = np.vstack([X1, X2])

gmm = GaussianMixtureModel(n_components=2)
gmm.fit(X_data)
probs = gmm.predict_proba(X_data)
print(f"前 5 个样本属于集群 0 的概率: {probs[:5, 0]}")

```

EOF

cat << 'EOF' > "${OUTPUT_DIR}/03_Probability_and_Stochastic_Processes/02_Stochastic_Processes_and_Markov_Chains.md"

# 第二章：随机过程基础、马尔可夫链与 MCMC 采样

## 1. 核心概念与数学表达

### 1.1 随机过程 (Stochastic Processes)

随机过程是定义在同一概率空间上的一族随机变量 $\{X(t), t \in T\}$。

* 若 index $T$ 为离散集（如 $\mathbb{N}$），称为**离散时间随机过程** $X_0, X_1, X_2, \dots$。
* **平稳随机过程 (Stationary Process)**：
* **严平稳**：任意有限维联合分布在时间平移下保持不变。
* **宽平稳 (Weakly/Wide-Sense Stationary, WSS)**：
1. 均值函数为常数：$\mathbb{E}[X(t)] = \mu$。
2. 自协方差函数只与时间差 $\tau$ 有关：$\gamma(t, t+\tau) = \mathbb{E}[(X(t)-\mu)(X(t+\tau)-\mu)] = R(\tau)$。





### 1.2 马尔可夫链 (Markov Chains)

满足无记忆性（马尔可夫性）的随机过程：


$$P(X_{n+1} = x_{n+1} \mid X_n = x_n, X_{n-1} = x_{n-1}, \dots, X_0 = x_0) = P(X_{n+1} = x_{n+1} \mid X_n = x_n)$$

* **转移概率矩阵 (Transition Matrix $P$)**：$P_{ij} = P(X_{n+1} = j \mid X_n = i)$，满足 $\sum_j P_{ij} = 1$。
* **平稳分布 (Stationary Distribution $\pi$)**：若概率向量 $\pi$ 满足：

$$\pi P = \pi, \quad \sum_i \pi_i = 1$$



则称 $\pi$ 为马尔可夫链的平稳分布。
* **细致平衡条件 (Detailed Balance Condition)**：若对所有状态 $i, j$，存在分布 $\pi$ 使得：

$$\pi_i P_{ij} = \pi_j P_{ji}$$



满足细致平衡条件的分布 $\pi$ 必然是该马尔可夫链的平稳分布（充分条件）。

---

## 2. 马尔可夫链蒙特卡洛采样 (MCMC)

### 2.1 Metropolis-Hastings (MH) 算法原理

为了从复杂的未归一化目标分布 $p(x) = \frac{\tilde{p}(x)}{Z}$ 中采样，引入提议分布 (Proposal Distribution) $q(x' \mid x)$。
为了满足细致平衡条件 $\pi(x) P(x \to x') = \pi(x') P(x' \to x)$，构造转移概率 $P(x \to x') = q(x' \mid x) \alpha(x, x')$，其中 $\alpha(x, x')$ 为**接受率 (Acceptance Rate)**：


$$\alpha(x, x') = \min \left( 1, \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)} \right) = \min \left( 1, \frac{\tilde{p}(x') q(x \mid x')}{\tilde{p}(x) q(x' \mid x)} \right)$$

### 2.2 吉布斯采样 (Gibbs Sampling)

吉布斯采样是 MH 算法的特例。在多维随机变量 $X = (X_1, X_2, \dots, X_d)$ 中，每次固定其他维度，依次从满条件分布 (Full Conditional Distribution) 中采样：


$$X_i^{(t+1)} \sim P(X_i \mid X_1^{(t+1)}, \dots, X_{i-1}^{(t+1)}, X_{i+1}^{(t)}, \dots, X_d^{(t)})$$


其接受率 $\alpha \equiv 1$（必定接受）。

---

## 3. AI/ML 经典应用案例：从复杂非标准双峰分布中采样的 Metropolis-Hastings 算法 Python 实现

### 3.1 目标分布定义

定义未归一化的双峰目标密度函数：


$$\tilde{p}(x) = \exp(-x^2) + 0.5 \exp(-(x - 4)^2)$$

### 3.2 Metropolis-Hastings 采样器完整代码

```python
import numpy as np

def target_density(x):
    # 未归一化的双峰高斯混合分布
    return np.exp(-x**2) + 0.5 * np.exp(-(x - 4.0)**2)

def metropolis_hastings_sampler(n_samples=10000, proposal_std=1.0):
    samples = np.zeros(n_samples)
    current_x = 0.0 # 初始状态
    accepted_count = 0
    
    for i in range(n_samples):
        # 1. 从对称提议分布 q(x' | x) = N(x, proposal_std^2) 生成候选样本
        proposed_x = np.random.normal(current_x, proposal_std)
        
        # 2. 计算接受率 alpha (因为 q 是对称的，q(x|x') = q(x'|x)，故可约简)
        p_current = target_density(current_x)
        p_proposed = target_density(proposed_x)
        
        alpha = min(1.0, p_proposed / p_current)
        
        # 3. 接受/拒绝判定
        if np.random.rand() < alpha:
            current_x = proposed_x
            accepted_count += 1
            
        samples[i] = current_x
        
    print(f"采样完成。接受率: {accepted_count / n_samples * 100:.2f}%")
    return samples

# 执行 MCMC 采样
np.random.seed(42)
mcmc_samples = metropolis_hastings_sampler(n_samples=20000, proposal_std=2.0)

# 丢弃 Burn-in 前预热阶段的样本
burn_in = 2000
final_samples = mcmc_samples[burn_in:]

print(f"采样均值: {np.mean(final_samples):.4f}")
print(f"采样方差: {np.var(final_samples):.4f}")

```

EOF

cat << 'EOF' > "${OUTPUT_DIR}/03_Probability_and_Stochastic_Processes/03_Information_Theory_and_Generative_Models.md"

# 第三章：信息论基础、KL散度与生成模型 (VAE/扩散模型) 数学根基

## 1. 信息论核心概念

### 1.1 熵、交叉熵与 KL 散度

* **香农熵 (Shannon Entropy)**：衡量随机变量不确定性：

$$H(X) = -\sum_{x} P(x) \log_2 P(x) \quad (\text{连续变量为微分熵 } h(X) = -\int p(x) \ln p(x) dx)$$


* **相对熵 / KL 散度 (Kullback-Leibler Divergence)**：衡量两个概率分布 $P$ 与 $Q$ 之间的差异：

$$D_{\text{KL}}(P \parallel Q) = \int p(x) \ln \frac{p(x)}{q(x)} dx = \mathbb{E}_{x \sim P} \left[ \ln p(x) - \ln q(x) \right]$$


* 性质：$D_{\text{KL}}(P \parallel Q) \ge 0$（由 Gibbs 不等式/Jansen 不等式保证）；不对称，即 $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$。


* **交叉熵 (Cross Entropy)**：

$$H(P, Q) = -\mathbb{E}_{x \sim P} [\ln q(x)] = H(P) + D_{\text{KL}}(P \parallel Q)$$



---

## 2. 变分自编码器 (VAE) 的变分下界 (ELBO) 数学推导

给定观测数据 $x$，欲最大化对数边际似然 $\ln p_\theta(x)$。引入关于隐变量 $z$ 的变分后验分布 $q_\phi(z \mid x)$。

### 2.1 ELBO 恒等式推导

$$\begin{aligned}
\ln p_\theta(x) &= \int q_\phi(z \mid x) \ln p_\theta(x) dz \\
&= \int q_\phi(z \mid x) \ln \frac{p_\theta(x, z)}{p_\theta(z \mid x)} dz \\
&= \int q_\phi(z \mid x) \ln \left( \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right) dz \\
&= \int q_\phi(z \mid x) \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} dz + \int q_\phi(z \mid x) \ln \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} dz \\
&= \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{\text{KL}}\left( q_\phi(z \mid x) \parallel p_\theta(z \mid x) \right)
\end{aligned}$$

由于 KL 散度 $D_{\text{KL}} \ge 0$，故对数似然可被证据下界 (Evidence Lower Bound, ELBO) 控底：


$$\ln p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$$

### 2.2 ELBO 拆解与重参数化技巧 (Reparameterization Trick)

将 ELBO 展开为重构项与正则化项：


$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z \mid x)} [\ln p_\theta(x \mid z)] - D_{\text{KL}}\left( q_\phi(z \mid x) \parallel p(z) \right)$$


假设标准高斯先验 $p(z) = \mathcal{N}(0, I)$ 且变分后验 $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$：

* **高斯 KL 散度解析解**：

$$D_{\text{KL}}\left( \mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I) \right) = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$


* **重参数化技巧**：直接采样 $z \sim \mathcal{N}(\mu, \sigma^2)$ 无法对 $\phi$ 求导。引入独立噪声 $\epsilon \sim \mathcal{N}(0, I)$，构造确定性变换：

$$z = \mu(x) + \sigma(x) \odot \epsilon$$



使得梯度可以通过 $z$ 顺利反向传播回编码器参数 $\phi$。

---

## 3. AI/ML 经典应用案例：变分自编码器 (VAE) 完整 PyTorch 架构实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        
        # 编码器网络 (Encoder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)   # 预测均值 mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # 预测对数方差 log(sigma^2)
        
        # 解码器网络 (Decoder)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # 1. 重构损失 (BCE / Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL 散度正则化项解析解
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# 验证 VAE 模块
if __name__ == "__main__":
    dummy_x = torch.rand(32, 1, 28, 28) # Batch 的 MNIST 伪图像
    vae = VariationalAutoencoder()
    
    recon_x, mu, logvar = vae(dummy_x)
    loss = vae_loss_function(recon_x, dummy_x, mu, logvar)
    
    print(f"输入图像张量: {dummy_x.shape}")
    print(f"重构图像张量: {recon_x.shape}")
    print(f"隐变量 Space Dimensions: {mu.shape}")
    print(f"计算所得总 Loss (BCE + KLD): {loss.item():.4f}")

```

EOF

# 设置文件可执行权限（若需要）

chmod +x "${OUTPUT_DIR}"/*/*.md

echo "=="
echo " 课程体系全套生成完毕！"
echo " 文件包含完整的专业理论推导与可执行 Python 算法代码。"
echo "=="
EOF

# 赋予 Bash 脚本执行权限并提示

chmod +x build_math_docs.sh

echo "Bash 脚本 'build_math_docs.sh' 已成功生成！"
echo "请在终端中运行以下命令以一键建卷并生成全套顶级教学文档："
echo "  ./build_math_docs.sh"

```
