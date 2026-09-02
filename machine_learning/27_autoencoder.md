# 自编码器 (Autoencoder, AE)

## 1. 算法原理

自编码器（Autoencoder）是一种无监督神经网络模型，主要用于数据的特征降维、去噪和特征表示学习。

自编码器由两部分组成：

1. **编码器 (Encoder)**：将高维输入数据 $x$ 映射/压缩到低维的隐空间（Latent Space）代码 $z$。
2. **解码器 (Decoder)**：将低维隐空间代码 $z$ 重构恢复为高维原始数据 $\hat{x}$。

---

## 2. 数学公式与推导

1. **编码过程**：

$$z = f_\theta(x) = \sigma(W_e x + b_e)$$

* z: 低维隐空间代码（Latent Space Vector）
* $f_\theta$: 编码器映射函数，由参数 $\theta$ 参数化
* $\theta(西塔)$: 编码器的参数集合
* x: 高维输入数据向量
* $\sigma(西格玛)$: 激活函数（如 Sigmoid、ReLU 等）
* $W_e$: 编码器的权重矩阵（Weight matrix of Encoder）
* $b_e$: 编码器的偏置向量（Bias vector of Encoder）

2. **解码过程**：

$$\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)$$

* $\hat{x}$: 重构的输入数据向量
* $g_\phi$: 解码器映射函数，由参数 $\phi$ 参数化
* $\phi(斐/斐尔)$: 解码器的参数集合
* z: 低维隐空间代码
* $\sigma(西格玛)$: 激活函数（如 Sigmoid、ReLU 等）
* $W_d$: 解码器的权重矩阵（Weight matrix of Decoder）
* $b_d$: 解码器的偏置向量（Bias vector of Decoder）

3. **损失函数 (MSE Loss)**：

$$L(x, \hat{x}) = \frac{1}{2} \Vert{}x - \hat{x}\Vert{}_2^2$$

* $L(x, \hat{x})$: 损失函数值（重构误差/损失）
* x: 原始高维输入数据向量
* $\hat{x}$: 解码重构的输出向量
* $\Vert{}\cdot\Vert{}_2$: $L_2$ 范数（欧几里得距离/欧氏范数）
* $\Vert{}\cdot\Vert{}_2^2$: $L_2$ 范数的平方（均方误差的基础计算部分）

---

## 3. ASCII 结构图

```
 [ 高维输入 x ]  --->  [ 编码器 Encoder ]  --->  [ 瓶颈层 Latent z ]
                                                         |
 [ 重构输出 x_hat ] <--  [ 解码器 Decoder ]  <-----------+


```

---

## 4. Python 代码实现 (基于 PyTorch)

```python
import torch
import torch.nn as nn
import torch.optim as optim

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

if __name__ == "__main__":
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    
    model = Autoencoder(input_dim=10, latent_dim=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()

    print(f"训练 100 轮后重构 Loss: {loss.item():.4f}")


```