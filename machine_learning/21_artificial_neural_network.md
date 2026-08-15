# 人工神经网络 (Artificial Neural Network, ANN)

## 1. 算法原理

人工神经网络（ANN）是受生物神经元网络启发的计算模型。最基础的结构是多层感知机（Multilayer Perceptron, MLP），由输入层、一个或多个隐藏层以及输出层构成。

每一层包含多个神经元，层与层之间通过权重 $W$ 和偏置 $b$ 连接。输入信号在网络中向后传播（前向传播），产生预测值；通过损失函数计算预测值与真实值之间的差距，再利用反向传播算法（Backpropagation）结合梯度下降（Gradient Descent）更新权重和偏置，从而实现模型的学习。

---

## 2. 数学公式与推导

现以单隐藏层的感知机为例进行推导：

### 2.1 前向传播 (Forward Propagation)
假设输入向量为 $\mathbf{x} \in \mathbb{R}^{d}$，隐藏层有 $m$ 个神经元，输出层有 $k$ 个神经元。

1. **隐藏层输入与激活**：
   $$\mathbf{z}^{(1)} = \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$
   $$\mathbf{a}^{(1)} = \sigma(\mathbf{z}^{(1)})$$
   其中 $\mathbf{W}^{(1)} \in \mathbb{R}^{m \times d}$，$\sigma(\cdot)$ 为非线性激活函数（如 Sigmoid、ReLU）。

2. **输出层输入与激活**：
   $$\mathbf{z}^{(2)} = \mathbf{W}^{(2)} \mathbf{a}^{(1)} + \mathbf{b}^{(2)}$$
   $$\hat{\mathbf{y}} = g(\mathbf{z}^{(2)})$$
   其中 $g(\cdot)$ 为输出层激活函数（如用于分类的 Softmax，或用于回归的恒等函数）。

### 2.2 损失函数 (Loss Function)
对于均方误差 (MSE) 损失：
$$L(\mathbf{W}, \mathbf{b}) = \frac{1}{2} \|\hat{\mathbf{y}} - \mathbf{y}\|_2^2$$

### 2.3 反向传播与链式法则 (Backpropagation)
根据链式法则计算损失函数对权重和偏置的偏导数：

1. **输出层梯度**：
   $$\delta^{(2)} = \frac{\partial L}{\partial \mathbf{z}^{(2)}} = (\hat{\mathbf{y}} - \mathbf{y}) \odot g'(\mathbf{z}^{(2)})$$
   $$\frac{\partial L}{\partial \mathbf{W}^{(2)}} = \delta^{(2)} (\mathbf{a}^{(1)})^T, \quad \frac{\partial L}{\partial \mathbf{b}^{(2)}} = \delta^{(2)}$$

2. **隐藏层梯度**：
   $$\delta^{(1)} = \frac{\partial L}{\partial \mathbf{z}^{(1)}} = \left( (\mathbf{W}^{(2)})^T \delta^{(2)} \right) \odot \sigma'(\mathbf{z}^{(1)})$$
   $$\frac{\partial L}{\partial \mathbf{W}^{(1)}} = \delta^{(1)} \mathbf{x}^T, \quad \frac{\partial L}{\partial \mathbf{b}^{(1)}} = \delta^{(1)}$$

3. **参数更新**：
   $$\mathbf{W}^{(l)} \leftarrow \mathbf{W}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{W}^{(l)}}$$
   $$\mathbf{b}^{(l)} \leftarrow \mathbf{b}^{(l)} - \eta \frac{\partial L}{\partial \mathbf{b}^{(l)}}$$
   其中 $\eta$ 为学习率。

---

## 3. ASCII 结构图


```

```
  [ 输入层 ]             [ 隐藏层 ]             [ 输出层 ]

x1 \                   /-- (a1) --\
    \---> (z1) -> σ --/            \---> (z1_out) -> g ---> y1_hat
x2 -----> (z2) -> σ ------> (a2) ------> (z2_out) -> g ---> y2_hat
    /---> (z3) -> σ --\            /
x3 /                   \-- (a3) --/

     |               |              |               |
     +--- W(1), b(1) +--------------+--- W(2), b(2) +

```

```

---

## 4. Python 代码实现 (基于 NumPy / Scikit-Learn)

### 4.1 NumPy 从零实现多层感知机 (MLP)

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1.0 - s)

class MLPFromScratch:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.1):
        self.lr = lr
        # 初始化权重与偏置
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, X):
        # X: (input_dim, batch_size)
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, Y):
        m = X.shape[1]
        
        # 计算输出层误差
        dz2 = self.a2 - Y
        dW2 = (1 / m) * np.dot(dz2, self.a1.T)
        db2 = (1 / m) * np.sum(dz2, axis=1, keepdims=True)

        # 计算隐藏层误差
        dz1 = np.dot(self.W2.T, dz2) * sigmoid_derivative(self.z1)
        dW1 = (1 / m) * np.dot(dz1, X.T)
        db1 = (1 / m) * np.sum(dz1, axis=1, keepdims=True)

        # 参数更新
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            self.forward(X)
            self.backward(X, Y)

# 示例：XOR 问题
if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]]) # 2x4
    Y = np.array([[0, 1, 1, 0]]) # 1x4

    mlp = MLPFromScratch(input_dim=2, hidden_dim=4, output_dim=1, lr=1.0)
    mlp.fit(X, Y, epochs=5000)

    preds = mlp.forward(X)
    print("XOR 预测结果:", preds.round(3))

```

### 4.2 Scikit-Learn 实现

```python
from sklearn.neural_network import MLPClassifier
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 0])

clf = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='lbfgs', max_iter=1000)
clf.fit(X, y)

print("Scikit-Learn XOR 预测:", clf.predict(X))

```

