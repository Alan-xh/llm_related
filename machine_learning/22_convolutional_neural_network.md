# 卷积神经网络 (Convolutional Neural Network, CNN)

## 1. 算法原理

卷积神经网络（CNN）是专门设计用于处理具有类似网格结构数据（如二维图像）的深度学习架构。其核心思想是**局部感受野（Local Receptive Field）**、**权值共享（Weight Sharing）** 以及 **下采样/池化（Pooling）**。

1. **卷积层 (Convolutional Layer)**：使用若干可学习的特征滤镜（Kernel/Filter）在输入特征图上滑动，通过局部点积提取空间局部特征（如边缘、纹理等）。
2. **激活函数 (Activation)**：通常使用 ReLU 函数引入非线性变化。
3. **池化层 (Pooling Layer)**：下采样操作（如最大池化 Max Pooling、平均池化 Average Pooling），降低特征图尺寸，减少参数量，同时增强特征的平移不变性。
4. **全连接层 (Fully Connected Layer)**：将二维特征图拉平后接入分类或回归网络。

---

## 2. 数学公式与推导

### 2.1 二维卷积操作 (2D Convolution)

假设输入特征图为 $X \in \mathbb{R}^{H \times W}$，卷积核为 $K \in \mathbb{R}^{k_h \times k_w}$，偏置为 $b$。

* X: 输入特征图（二维张量/矩阵）
* $\mathbb{R}$: 实数集
* H: 输入特征图的高度（行数）
* W: 输入特征图的宽度（列数）
* K: 卷积核/滤镜矩阵
* $k_h$: 卷积核的高度
* $k_w$: 卷积核的宽度
* b: 偏置项（标量）

输出特征图 $Y$ 的坐标 $(i, j)$ 处的元素计算公式为：

$$Y_{i, j} = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} X_{i+m, j+n} \cdot K_{m, n} + b$$

* $Y_{i, j}$: 输出特征图在第 $i$ 行、第 $j$ 列位置的像素值
* $\sum$: 求和符号
* $k_h$: 卷积核的高度
* $k_w$: 卷积核的宽度
* m: 卷积核高度方向上的索引/偏移量（取值范围 $0$ 到 $k_h-1$）
* n: 卷积核宽度方向上的索引/偏移量（取值范围 $0$ 到 $k_w-1$）
* $X_{i+m, j+n}$: 输入特征图中对应感受野窗口内 $(i+m, j+n)$ 位置的像素值
* $K_{m, n}$: 卷积核内第 $m$ 行、第 $n$ 列的权重参数
* b: 偏置项

特征图尺寸变换规则：

$$H_{out} = \left\lfloor \frac{H_{in} - k_h + 2P}{S} \right\rfloor + 1, \quad W_{out} = \left\lfloor \frac{W_{in} - k_w + 2P}{S} \right\rfloor + 1$$

* $H_{out}$: 卷积输出特征图的高度
* $W_{out}$: 卷积输出特征图的宽度
* $\lfloor \cdot \rfloor$: 向下取整符号
* $H_{in}$: 输入特征图的高度
* $W_{in}$: 输入特征图的宽度
* $k_h$: 卷积核的高度
* $k_w$: 卷积核的宽度
* P: 填充（Padding）像素数
* S: 滑动步长（Stride）

其中 $P$ 为填充（Padding），$S$ 为步长（Stride）。

### 2.2 最大池化 (Max Pooling)

在尺寸为 $p_h \times p_w$ 的窗口内提取最大值：

$$Y_{i, j} = \max_{0 \le m < p_h, 0 \le n < p_w} X_{i \cdot S + m, j \cdot S + n}$$

* $Y_{i, j}$: 池化输出特征图在第 $i$ 行、第 $j$ 列位置的元素
* $\max$: 求最大值函数
* $p_h$: 池化窗口的高度
* $p_w$: 池化窗口的宽度
* m: 池化窗口高度方向上的索引（取值范围 $0$ 到 $p_h-1$）
* n: 池化窗口宽度方向上的索引（取值范围 $0$ 到 $p_w-1$）
* $X_{i \cdot S + m, j \cdot S + n}$: 输入特征图中处于当前池化窗口内的像素值
* S: 池化滑动的步长（Stride）

### 2.3 反向传播 (Backpropagation in Conv Layer)

对于卷积核权重 $K_{m,n}$ 的梯度更新，根据链式法则：

$$\frac{\partial L}{\partial K_{m,n}} = \sum_{i} \sum_{j} \frac{\partial L}{\partial Y_{i,j}} \cdot X_{i+m, j+n}$$

* $\frac{\partial L}{\partial K_{m,n}}$: 损失函数 $L$ 关于卷积核第 $m$ 行第 $n$ 列权重 $K_{m,n}$ 的偏导数（梯度）
* $\partial(\text{德尔塔})$: 偏导数符号
* L: 损失函数（Loss Function）
* $K_{m,n}$: 卷积核内第 $m$ 行第 $n$ 列的权重
* $\sum$: 求和符号
* i: 卷积输出特征图的行索引
* j: 卷积输出特征图的列索引
* $\frac{\partial L}{\partial Y_{i,j}}$: 损失函数 $L$ 关于输出特征图第 $i$ 行第 $j$ 列元素的偏导数（上游梯度）
* $Y_{i,j}$: 卷积输出特征图在 $(i,j)$ 位置的值
* $X_{i+m, j+n}$: 正向传播时与权重 $K_{m,n}$ 相乘的前向输入特征图元素

---

## 3. ASCII 结构图

```
 [输入图像 28x28] ---> [卷积层 + ReLU] ---> [最大池化层] ---> [全连接层] ---> [输出]
 
 +---+---+---+         +---+---+               +---+
 |   |   |   | * [3x3] |   |   |    [2x2]      |   |        [Softmax]
 +---+---+---+  -----> +---+---+   ------->    +---+    --->  ( 0.9 )
 |   |   |   |  Kernel |   |   |   Max Pool    |   |          ( 0.1 )
 +---+---+---+         +---+---+               +---+


```

---

## 4. Python 代码实现 (基于 NumPy)

### 4.1 NumPy 从零实现 2D 卷积与最大池化

```python
import numpy as np

def conv2d_forward(X, K, b, stride=1, padding=0):
    if padding > 0:
        X_padded = np.pad(X, ((padding, padding), (padding, padding)), mode='constant')
    else:
        X_padded = X
    
    H_in, W_in = X_padded.shape
    kh, kw = K.shape
    
    H_out = (H_in - kh) // stride + 1
    W_out = (W_in - kw) // stride + 1
    
    output = np.zeros((H_out, W_out))
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            h_end = h_start + kh
            w_start = j * stride
            w_end = w_start + kw
            
            patch = X_padded[h_start:h_end, w_start:w_end]
            output[i, j] = np.sum(patch * K) + b
            
    return output

def max_pooling_forward(X, pool_size=2, stride=2):
    H_in, W_in = X.shape
    H_out = (H_in - pool_size) // stride + 1
    W_out = (W_in - pool_size) // stride + 1
    
    output = np.zeros((H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            patch = X[i*stride : i*stride+pool_size, j*stride : j*stride+pool_size]
            output[i, j] = np.max(patch)
            
    return output

# 运行测试
if __name__ == "__main__":
    np.random.seed(42)
    image = np.random.randn(6, 6)
    kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])
    
    conv_res = conv2d_forward(image, kernel, b=0, stride=1, padding=1)
    pool_res = max_pooling_forward(conv_res, pool_size=2, stride=2)
    
    print("卷积后特征图形状:", conv_res.shape)
    print("池化后特征图形状:", pool_res.shape)


```