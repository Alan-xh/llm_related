
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


2. **解码过程**：

$$\hat{x} = g_\phi(z) = \sigma(W_d z + b_d)$$


3. **损失函数 (MSE Loss)**：

$$L(x, \hat{x}) = \frac{1}{2} \|x - \hat{x}\|_2^2$$



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

