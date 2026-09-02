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

* $\mathcal{L}^{(t)}$: 第 $t$ 轮迭代时的整体目标函数值
* $t$: 当前迭代的轮数/树的序号
* $n$: 样本总数量
* $i$: 样本的索引编号
* $l$: 针对单个样本的损失函数 (Loss Function)
* $y_i$: 第 $i$ 个样本的真实标签值
* $\hat{y}_i^{(t-1)}$: 前 $t-1$ 棵树对第 $i$ 个样本的累积预测值
* $f_t$: 第 $t$ 棵需要学习的新决策树模型
* $\mathbf{x}_i$: 第 $i$ 个样本的特征向量
* $\Omega(\欧米伽)$: 树的正则化项，用于控制树的复杂度，防止过拟合

采用二阶泰勒展开：

$$l(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)) \approx l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(\mathbf{x}_i) + \frac{1}{2} h_i f_t^2(\mathbf{x}_i)$$

* $l$: 针对单个样本的损失函数
* $y_i$: 第 $i$ 个样本的真实标签值
* $\hat{y}_i^{(t-1)}$: 前 $t-1$ 棵树对第 $i$ 个样本的累积预测值
* $f_t(\mathbf{x}_i)$: 第 $t$ 棵树对第 $i$ 个样本特征向量 $\mathbf{x}_i$ 的预测输出值
* $g_i$: 损失函数在当前预测值处的一阶导数（一阶梯度）
* $h_i$: 损失函数在当前预测值处的二阶导数（二阶梯度）

其中：

$$g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}$$

* $g_i$: 针对第 $i$ 个样本的损失函数一阶导数（一阶梯度）
* $h_i$: 针对第 $i$ 个样本的损失函数二阶导数（二阶梯度）
* $l$: 损失函数
* $y_i$: 第 $i$ 个样本的真实标签值
* $\hat{y}^{(t-1)}$: 前 $t-1$ 步对样本的累积预测值
* $\partial$: 偏微分符号

### 2.2 正则化与叶子节点权重求解

树复杂度定义：

$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$

* $\Omega(\欧米伽)$: 第 $t$ 棵树 $f_t$ 的结构复杂度正则化惩罚项
* $f_t$: 第 $t$ 棵决策树模型
* $\gamma(\伽马)$: 控制叶节点数量的惩罚系数（树结构正则化参数，用于控制分裂阈值）
* $T$: 该决策树中的叶子节点总数量
* $\lambda(\兰姆达)$: 叶节点权重的 L2 正则化系数
* $j$: 叶子节点的索引编号
* $w_j$: 第 $j$ 个叶子节点的预测权重值（叶节点得分）

精简后的目标函数：

$$\tilde{\mathcal{L}}^{(t)} = \sum_{j=1}^{T} \left[ \left(\sum_{i \in I_j} g_i\right) w_j + \frac{1}{2} \left(\sum_{i \in I_j} h_i + \lambda\right) w_j^2 \right] + \gamma T$$

* $\tilde{\mathcal{L}}^{(t)}$: 移除常数项后简化得到的第 $t$ 步目标函数近似值
* $T$: 叶子节点总数量
* $j$: 叶子节点的索引编号
* $I_j$: 被划分到第 $j$ 个叶子节点上的样本索引集合
* $i$: 归属于 $I_j$ 的样本索引
* $g_i$: 第 $i$ 个样本的一阶梯度
* $w_j$: 第 $j$ 个叶子节点的预测权重值
* $h_i$: 第 $i$ 个样本的二阶梯度
* $\lambda(\兰姆达)$: 叶节点权重的 L2 正则化系数
* $\gamma(\伽马)$: 叶节点数量的惩罚系数

最优叶节点权重 $w_j^*$ 及得分：

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \text{Score} = -\frac{1}{2} \sum_{j=1}^{T} \frac{(\sum_{i \in I_j} g_i)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T$$

* $w_j^*$: 第 $j$ 个叶子节点使得目标函数最小化时的最优权重值
* $I_j$: 划分至第 $j$ 个叶子节点上的样本集合
* $i$: 样本索引
* $g_i$: 第 $i$ 个样本的一阶梯度
* $h_i$: 第 $i$ 个样本的二阶梯度
* $\lambda(\兰姆达)$: 叶节点权重的 L2 正则化系数
* $\text{Score}$: 评价整棵树结构优劣的综合结构得分（代入最优权重后的目标函数极小值，值越小越好）
* $j$: 叶子节点索引
* $T$: 叶子节点总数
* $\gamma(\伽马)$: 叶节点数量的惩罚系数

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