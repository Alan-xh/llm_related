# 逻辑回归 (Logistic Regression)

## 1. 算法原理与概述

逻辑回归（Logistic Regression）虽名为回归，但本质上是一种解决**二分类问题**的监督学习算法。它通过 Sigmoid 函数将线性回归的连续输出映射到 $(0, 1)$ 区间，从而将其解释为事件发生的概率。

```
+-------------------+       +---------------+       +------------------+       +---------------+
|  输入特征矩阵 X   | ----> |  z = W^T*X+b  | ----> | Sigmoid: 1/(1+e^-z) | ----> | 预测概率 P(y=1) |
+-------------------+       +---------------+       +------------------+       +---------------+
                                                                                       |
                                                                                       v
                                                                               +---------------+
                                                                               | 交叉熵损失 LogLoss|
                                                                               +---------------+

```

---

## 2. 数学原理与推导

### 2.1 Sigmoid 函数

Sigmoid 激活函数表达式：


$$\sigma(z) = \frac{1}{1 + e^{-z}}$$


定义线性组合 $z = \mathbf{w}^T \mathbf{x} + b$，则正例的预测概率为：


$$P(y=1\vert{}\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$


负例概率为 $P(y=0\vert{}\mathbf{x}) = 1 - P(y=1\vert{}\mathbf{x})$。

### 2.2 损失函数 (对数损失 / 交叉熵损失)

采用极大似然估计推导得到二分类交叉熵损失函数（Binary Cross-Entropy Loss）：


$$J(\mathbf{w}, b) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

### 2.3 梯度推导

对参数 $\mathbf{w}$ 求偏导：


$$\frac{\partial J}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$


参数更新规则：


$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \frac{1}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})$$

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score, classification_report

class CustomLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            dw = (1 / m) * np.dot(X.T, (y_predicted - y))
            db = (1 / m) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        y_predicted_cls = [1 if i >= threshold else 0 for i in self.predict_proba(X)]
        return np.array(y_predicted_cls)

# 测试对比
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=4, random_state=42)

    custom_lr = CustomLogisticRegression(lr=0.1, n_iters=1000)
    custom_lr.fit(X, y)
    preds_custom = custom_lr.predict(X)

    sk_lr = SklearnLogisticRegression()
    sk_lr.fit(X, y)
    preds_sk = sk_lr.predict(X)

    print("Custom Logistic Regression Acc:", accuracy_score(y, preds_custom))
    print("Sklearn Logistic Regression Acc:", accuracy_score(y, preds_sk))

```

