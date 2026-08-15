# XGBoost (Extreme Gradient Boosting)

## 1. 算法原理与概述

XGBoost 是对传统 GBDT 的高效扩展实现。它在损失函数中引入了二阶泰勒展开，并增加了控制树复杂度的正则化项，结合特征并行与列抽样，实现了更高的精度与极佳的计算速度。

```
 [ 目标函数 ] = [ 二阶泰勒展开拟合损失 (g_i, h_i) ] + [ 结构正则化 (树深度, 叶节点数, 权重L2) ]
                                          |
                                          v
                              [ 精确/近似节点分裂优化 ]

```

---

## 2. 数学原理

### 2.1 目标函数与二阶泰勒展开

在第 $t$ 步，目标函数为：


$$\mathcal{L}^{(t)} = \sum_{i=1}^{n} l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) + \Omega(f_t)$$

采用二阶泰勒展开：


$$l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)$$


其中：


$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$

### 2.2 正则化与叶子节点权重求解

树复杂度定义：


$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$


精简后的目标函数：


$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$

最优叶节点权重 $w_j^*$ 及得分：


$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \text{Score} = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

---

## 3. Python / NumPy / XGBoost 实现

```python
import numpy as np

# 简化演示: 手动计算 XGBoost 一阶与二阶梯度
def mse_gradients(y_true, y_pred):
    # 二次损失: L = 0.5 * (y - y_hat)^2
    g = y_pred - y_true
    h = np.ones_like(y_true)
    return g, h

class SimpleXGBoostNode:
    def __init__(self, reg_lambda=1.0, gamma=0.0):
        self.reg_lambda = reg_lambda
        self.gamma = gamma

    def compute_leaf_weight(self, g, h):
        return -np.sum(g) / (np.sum(h) + self.reg_lambda)

    def compute_split_gain(self, g_left, h_left, g_right, h_right):
        def score(G, H):
            return (G ** 2) / (H + self.reg_lambda)
        
        G_L, H_L = np.sum(g_left), np.sum(h_left)
        G_R, H_R = np.sum(g_right), np.sum(h_right)
        G_P, H_P = G_L + G_R, H_L + H_R

        gain = 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G_P, H_P)) - self.gamma
        return gain

if __name__ == "__main__":
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.2, 1.8, 3.3, 3.7])

    g, h = mse_gradients(y_true, y_pred)
    node = SimpleXGBoostNode(reg_lambda=1.0, gamma=0.1)

    leaf_w = node.compute_leaf_weight(g, h)
    print("XGBoost 算例一阶梯度 g:", g)
    print("XGBoost 算例二阶梯度 h:", h)
    print("最佳叶子节点更新权重:", leaf_w)

```

