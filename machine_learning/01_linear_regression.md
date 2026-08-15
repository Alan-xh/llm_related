# 线性回归 (Linear Regression)

## 1. 算法原理与概述
线性回归（Linear Regression）是一种监督学习算法，用于建立自变量（特征 $X$）与因变量（目标 $y$）之间的线性关系。其基本思想是找到最佳拟合直线或超平面，使得模型预测值与真实值之间的偏差最小化。

在实际应用中，线性回归常用于预测房价、销售额趋势分析、风险评估等连续数值回归任务。


```

+-------------------+       +--------------------+       +-------------------+
|  输入特征矩阵 X   | ----> | 线性变换 W^T * X+b  | ----> |  预测连续值 y_hat  |
+-------------------+       +--------------------+       +-------------------+
|
v
+------------------+
| 计算损失 (MSE)    |
+------------------+
|
v
+------------------+
| 梯度下降 / 最小二乘 |
+------------------+

```

---

## 2. 数学原理与推导

### 2.1 模型表达
假定样本集包含 $n$ 个特征，模型的表达形式为：
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b = \mathbf{w}^T \mathbf{x} + b$$

将偏置项 $b$ 合并入权重向量 $\mathbf{w}$，并在向量 $\mathbf{x}$ 中添加常数项 $1$，即：
$$\mathbf{x} = [1, x_1, x_2, \dots, x_n]^T, \quad \mathbf{w} = [b, w_1, w_2, \dots, w_n]^T$$
得到简化的矩阵形式：
$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w}$$

### 2.2 损失函数 (均方误差 MSE)
为了衡量预测值与真实值之间的差距，采用均方误差（Mean Squared Error, MSE）作为损失函数：
$$J(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)})^2 = \frac{1}{2m} (\mathbf{X}\mathbf{w} - \mathbf{y})^T (\mathbf{X}\mathbf{w} - \mathbf{y})$$

### 2.3 求解方法

#### A. 均方误差极小化（正规方程 / Normal Equation）
对权重向量 $\mathbf{w}$ 求导并令导数为零：
$$\frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = \frac{1}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w} - \mathbf{y}) = 0$$
$$\mathbf{X}^T \mathbf{X} \mathbf{w} = \mathbf{X}^T \mathbf{y}$$
当 $\mathbf{X}^T \mathbf{X}$ 可逆时，解析解为：
$$\mathbf{w}^* = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

#### B. 梯度下降法 (Gradient Descent)
当数据量庞大或矩阵不可逆时，使用梯度下降进行参数更新：
$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \frac{\partial J(\mathbf{w})}{\partial \mathbf{w}} = \mathbf{w}^{(t)} - \frac{\alpha}{m} \mathbf{X}^T (\mathbf{X}\mathbf{w}^{(t)} - \mathbf{y})$$
其中 $\alpha$ 为学习率。

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. NumPy 底层手动实现线性回归
class CustomLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(self, lr=0.01, n_iters=1000, method='gradient_descent'):
        self.lr = lr
        self.n_iters = n_iters
        self.method = method
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        m, n = X.shape
        if self.method == 'normal_equation':
            # 添加偏置项 1
            X_b = np.c_[np.ones((m, 1)), X]
            # 正规方程计算: w = (X^T * X)^(-1) * X^T * y
            theta = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
            self.bias = theta[0]
            self.weights = theta[1:]
        else:
            # 梯度下降法
            self.weights = np.zeros(n)
            self.bias = 0.0
            
            for _ in range(self.n_iters):
                y_pred = np.dot(X, self.weights) + self.bias
                dw = (1 / m) * np.dot(X.T, (y_pred - y))
                db = (1 / m) * np.sum(y_pred - y)
                
                self.weights -= self.lr * dw
                self.bias -= self.lr * db
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# 2. 验证与对比
if __name__ == "__main__":
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X.squeeze() + np.random.randn(100) * 0.5

    # NumPy 梯度下降模型
    custom_model = CustomLinearRegression(lr=0.1, n_iters=1000)
    custom_model.fit(X, y)
    y_pred_custom = custom_model.predict(X)

    # Scikit-Learn 官方模型
    sk_model = SklearnLinearRegression()
    sk_model.fit(X, y)
    y_pred_sk = sk_model.predict(X)

    print(f"Custom Model -> Weights: {custom_model.weights[0]:.4f}, Bias: {custom_model.bias:.4f}")
    print(f"Sklearn Model -> Weights: {sk_model.coef_[0]:.4f}, Bias: {sk_model.intercept_:.4f}")
    print(f"Custom MSE: {mean_squared_error(y, y_pred_custom):.4f}, R2: {r2_score(y, y_pred_custom):.4f}")
    print(f"Sklearn MSE: {mean_squared_error(y, y_pred_sk):.4f}, R2: {r2_score(y, y_pred_sk):.4f}")

```

