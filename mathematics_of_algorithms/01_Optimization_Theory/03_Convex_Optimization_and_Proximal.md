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

