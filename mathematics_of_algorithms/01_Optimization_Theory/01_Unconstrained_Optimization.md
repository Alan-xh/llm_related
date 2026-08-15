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

