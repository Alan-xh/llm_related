# 第三章：信息论基础、KL散度与生成模型 (VAE/扩散模型) 数学根基

## 1. 信息论核心概念

### 1.1 熵、交叉熵与 KL 散度

* **香农熵 (Shannon Entropy)**：衡量随机变量不确定性：

$$H(X) = -\sum_{x} P(x) \log_2 P(x) \quad (\text{连续变量为微分熵 } h(X) = -\int p(x) \ln p(x) dx)$$


* **相对熵 / KL 散度 (Kullback-Leibler Divergence)**：衡量两个概率分布 $P$ 与 $Q$ 之间的差异：

$$D_{\text{KL}}(P \parallel Q) = \int p(x) \ln \frac{p(x)}{q(x)} dx = \mathbb{E}_{x \sim P} \left[ \ln p(x) - \ln q(x) \right]$$


* 性质：$D_{\text{KL}}(P \parallel Q) \ge 0$（由 Gibbs 不等式/Jansen 不等式保证）；不对称，即 $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$。


* **交叉熵 (Cross Entropy)**：

$$H(P, Q) = -\mathbb{E}_{x \sim P} [\ln q(x)] = H(P) + D_{\text{KL}}(P \parallel Q)$$



---

## 2. 变分自编码器 (VAE) 的变分下界 (ELBO) 数学推导

给定观测数据 $x$，欲最大化对数边际似然 $\ln p_\theta(x)$。引入关于隐变量 $z$ 的变分后验分布 $q_\phi(z \mid x)$。

### 2.1 ELBO 恒等式推导

$$\begin{aligned}
\ln p_\theta(x) &= \int q_\phi(z \mid x) \ln p_\theta(x) dz \\
&= \int q_\phi(z \mid x) \ln \frac{p_\theta(x, z)}{p_\theta(z \mid x)} dz \\
&= \int q_\phi(z \mid x) \ln \left( \frac{p_\theta(x, z)}{q_\phi(z \mid x)} \cdot \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} \right) dz \\
&= \int q_\phi(z \mid x) \ln \frac{p_\theta(x, z)}{q_\phi(z \mid x)} dz + \int q_\phi(z \mid x) \ln \frac{q_\phi(z \mid x)}{p_\theta(z \mid x)} dz \\
&= \mathcal{L}_{\text{ELBO}}(\theta, \phi; x) + D_{\text{KL}}\left( q_\phi(z \mid x) \parallel p_\theta(z \mid x) \right)
\end{aligned}$$

由于 KL 散度 $D_{\text{KL}} \ge 0$，故对数似然可被证据下界 (Evidence Lower Bound, ELBO) 控底：


$$\ln p_\theta(x) \ge \mathcal{L}_{\text{ELBO}}(\theta, \phi; x)$$

### 2.2 ELBO 拆解与重参数化技巧 (Reparameterization Trick)

将 ELBO 展开为重构项与正则化项：


$$\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi(z \mid x)} [\ln p_\theta(x \mid z)] - D_{\text{KL}}\left( q_\phi(z \mid x) \parallel p(z) \right)$$


假设标准高斯先验 $p(z) = \mathcal{N}(0, I)$ 且变分后验 $q_\phi(z \mid x) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$：

* **高斯 KL 散度解析解**：

$$D_{\text{KL}}\left( \mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I) \right) = -\frac{1}{2} \sum_{j=1}^J \left( 1 + \ln(\sigma_j^2) - \mu_j^2 - \sigma_j^2 \right)$$


* **重参数化技巧**：直接采样 $z \sim \mathcal{N}(\mu, \sigma^2)$ 无法对 $\phi$ 求导。引入独立噪声 $\epsilon \sim \mathcal{N}(0, I)$，构造确定性变换：

$$z = \mu(x) + \sigma(x) \odot \epsilon$$



使得梯度可以通过 $z$ 顺利反向传播回编码器参数 $\phi$。

---

## 3. AI/ML 经典应用案例：变分自编码器 (VAE) 完整 PyTorch 架构实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VariationalAutoencoder, self).__init__()
        
        # 编码器网络 (Encoder)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)   # 预测均值 mu
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # 预测对数方差 log(sigma^2)
        
        # 解码器网络 (Decoder)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        # z = mu + std * epsilon
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    # 1. 重构损失 (BCE / Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # 2. KL 散度正则化项解析解
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD

# 验证 VAE 模块
if __name__ == "__main__":
    dummy_x = torch.rand(32, 1, 28, 28) # Batch 的 MNIST 伪图像
    vae = VariationalAutoencoder()
    
    recon_x, mu, logvar = vae(dummy_x)
    loss = vae_loss_function(recon_x, dummy_x, mu, logvar)
    
    print(f"输入图像张量: {dummy_x.shape}")
    print(f"重构图像张量: {recon_x.shape}")
    print(f"隐变量 Space Dimensions: {mu.shape}")
    print(f"计算所得总 Loss (BCE + KLD): {loss.item():.4f}")

```

