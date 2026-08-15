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

