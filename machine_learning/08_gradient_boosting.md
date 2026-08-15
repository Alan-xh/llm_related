# 梯度提升 (Gradient Boosting / GBDT)

## 1. 算法原理与概述

梯度提升树（GBDT）是一种串行 Boosting 集成学习方法。与 AdaBoost 改变样本权重不同，GBDT 的核心思想是**让下一棵决策树拟合前面所有树预测结果的负梯度（即残差）**。

```
 [ 输入 X ] ---> [ 初始树 F_0 ] ---> 算残差 r_1 = y - F_0(X)
                                            |
                                            v
                                 [ 树 T_1 拟合残差 r_1 ] ---> 更新 F_1 = F_0 + gamma * T_1
                                                                        |
                                                                        v
                                                             [ 树 T_2 拟合残差 r_2 ] ...

```

---

## 2. 数学原理

### 2.1 梯度提升算法步骤

定义损失函数为 $L(y, F(\mathbf{x}))$。

1. **初始化常数模型**：

$$F_0(\mathbf{x}) = \arg\min_\gamma \sum_{i=1}^{m} L(y_i, \gamma)$$


2. **对于 $m = 1$ 到 $M$ 棵树**：
a. **计算伪残差 (负梯度)**：

$$r_{im} = -\left[ \frac{\partial L(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F(\mathbf{x})=F_{m-1}(\mathbf{x})}$$



b. **对残差 fit 一棵决策树**，得到叶子区域 $R_{jm}$。
c. **更新集成模型**：

$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \sum_{j} \gamma_{jm} I(\mathbf{x} \in R_{jm})$$



对于均方损失 $L(y, F) = \frac{1}{2}(y - F)^2$，负梯度恰好即为标准残差 $y_i - F_{m-1}(\mathbf{x}_i)$。

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor as SklearnGBDT
from sklearn.metrics import mean_squared_error

class CustomGBDTRegressor:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        # 初始化基准预测值
        self.f0 = np.mean(y)
        F = np.full(shape=y.shape, fill_value=self.f0)

        for _ in range(self.n_estimators):
            # 均方损失下的残差
            residual = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            F += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict(self, X):
        F = np.full(shape=(X.shape[0],), fill_value=self.f0)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

if __name__ == "__main__":
    X = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = X.squeeze() ** 2 + np.random.normal(0, 1, size=100)

    gbdt = CustomGBDTRegressor(n_estimators=20, learning_rate=0.1, max_depth=3)
    gbdt.fit(X, y)
    print("Custom GBDT MSE:", mean_squared_error(y, gbdt.predict(X)))

    sk_gbdt = SklearnGBDT(n_estimators=20, learning_rate=0.1, max_depth=3, random_state=42)
    sk_gbdt.fit(X, y)
    print("Sklearn GBDT MSE:", mean_squared_error(y, sk_gbdt.predict(X)))

```

