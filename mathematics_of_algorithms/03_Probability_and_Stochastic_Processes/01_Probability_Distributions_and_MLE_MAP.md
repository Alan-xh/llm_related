# 第一章：核心概率分布、最大似然估计与贝叶斯后验估计

## 1. 核心概念与概率分布

### 1.1 概率空间与贝叶斯定理

概率空间由元组 $(\Omega, \mathcal{F}, P)$ 定义。
**贝叶斯定理 (Bayes' Theorem)**：


$$P(\theta \mid X) = \frac{P(X \mid \theta) P(\theta)}{P(X)} = \frac{P(X \mid \theta) P(\theta)}{\int_\Theta P(X \mid \theta') P(\theta') d\theta'}$$

* $P(\theta)$：**先验概率 (Prior)**
* $P(X \mid \theta)$：**似然函数 (Likelihood)**
* $P(\theta \mid X)$：**后验概率 (Posterior)**
* $P(X)$：**边际似然/证据 (Evidence)**

### 1.2 高维多元高斯分布 (Multivariate Gaussian Distribution)

向量 $X \in \mathbb{R}^d$ 服从多元正态分布 $X \sim \mathcal{N}(\mu, \Sigma)$，其概率密度函数 (PDF) 为：


$$p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{d/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (x - \mu)^T \Sigma^{-1} (x - \mu) \right)$$


其中 $\mu \in \mathbb{R}^d$ 为均值向量，$\Sigma \in \mathbb{R}^{d \times d}$ 为协方差矩阵 ($\Sigma \succeq 0$)。指数项中的二次型 $(x - \mu)^T \Sigma^{-1} (x - \mu)$ 称为**马氏距离 (Mahalanobis Distance)** 的平方。

### 1.3 共轭先验 (Conjugate Priors)

若后验分布 $P(\theta \mid X)$ 与先验分布 $P(\theta)$ 属于同一个函数族，则称该先验与似然共轭。

* **Beta-Binomial 共轭**：似然为二项分布，先验为 Beta 分布 $\text{Beta}(\alpha, \beta)$，后验为 $\text{Beta}(\alpha + k, \beta + n - k)$。
* **Dirichlet-Multinomial 共轭**：似然为多项分布，先验为 Dirichlet 分布（Beta 在多维的推广）。

---

## 2. 估计理论：MLE, MAP 与 贝叶斯估计

### 2.1 最大似然估计 (Maximum Likelihood Estimation, MLE)

假设数据 $X = \{x_1, \dots, x_N\}$ 独立同分布 (i.i.d.)：


$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ln P(X \mid \theta) = \arg\max_\theta \sum_{i=1}^N \ln P(x_i \mid \theta)$$

### 2.2 最大后验估计 (Maximum A Posteriori, MAP)

引入参数的先验分布 $P(\theta)$：


$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \ln P(\theta \mid X) = \arg\max_\theta \left[ \sum_{i=1}^N \ln P(x_i \mid \theta) + \ln P(\theta) \right]$$

**与机器学习损失函数的正则化等价性**：

* 若先验 $P(\theta) \sim \mathcal{N}(0, \sigma_0^2 I)$（高斯先验），$\ln P(\theta) \propto -\Vert{}\theta\Vert{}_2^2 \implies$ **$L_2$ 正则化 (Ridge)**。
* 若先验 $P(\theta) \sim \text{Laplace}(0, b)$（拉普拉斯先验），$\ln P(\theta) \propto -\Vert{}\theta\Vert{}_1 \implies$ **$L_1$ 正则化 (Lasso)**。

---

## 3. AI/ML 经典应用案例：高斯混合模型 (GMM) 与 EM (期望最大化) 算法推导及 Python 实现

### 3.1 GMM 概率生成模型

高斯混合模型假设数据由 $K$ 个高斯分布的混合生成：


$$p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k), \quad \sum_{k=1}^K \pi_k = 1$$


包含隐变量 $z_i \in \{1, \dots, K\}$ 表示样本 $x_i$ 来自第 $k$ 个高斯分量。

### 3.2 EM 算法迭代推导

* **E 步 (Expectation)**：计算隐变量的后验概率（响应度 $\gamma_{ik}$）：

$$\gamma_{ik} = P(z_i = k \mid x_i; \theta) = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}$$


* **M 步 (Maximization)**：最大化期望对数似然，更新参数：

$$N_k = \sum_{i=1}^N \gamma_{ik}$$


$$\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} x_i$$


$$\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^N \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T$$


$$\pi_k^{\text{new}} = \frac{N_k}{N}$$



### 3.3 高斯混合模型 EM 算法完整实现

```python
import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        N, D = X.shape
        np.random.seed(42)
        
        # 参数初始化
        self.pi = np.full(self.K, 1.0 / self.K)
        random_indices = np.random.choice(N, self.K, replace=False)
        self.mu = X[random_indices]
        self.Sigma = np.array([np.eye(D) for _ in range(self.K)])
        
        log_likelihood_old = 0
        
        for iteration in range(self.max_iter):
            # --- E 步: 计算响应度 gamma (N, K) ---
            gamma = np.zeros((N, self.K))
            for k in range(self.K):
                # 计算多元正态 PDF
                diff = X - self.mu[k]
                inv_Sigma = np.linalg.inv(self.Sigma[k] + 1e-6 * np.eye(D))
                det_Sigma = np.linalg.det(self.Sigma[k] + 1e-6 * np.eye(D))
                
                norm_const = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det_Sigma) + 1e-15)
                exon = -0.5 * np.sum(np.dot(diff, inv_Sigma) * diff, axis=1)
                gamma[:, k] = self.pi[k] * norm_const * np.exp(exon)
                
            sum_gamma = np.sum(gamma, axis=1, keepdims=True) + 1e-15
            gamma /= sum_gamma
            
            # --- M 步: 更新参数 ---
            N_k = np.sum(gamma, axis=0)
            
            for k in range(self.K):
                self.mu[k] = np.sum(gamma[:, k:], axis=0) / N_k[k] if False else np.sum(gamma[:, [k]] * X, axis=0) / N_k[k]
                diff = X - self.mu[k]
                self.Sigma[k] = np.dot((gamma[:, [k]] * diff).T, diff) / N_k[k]
                self.pi[k] = N_k[k] / N
                
            # 计算对数似然
            log_likelihood = np.sum(np.log(sum_gamma))
            if abs(log_likelihood - log_likelihood_old) < self.tol:
                print(f"GMM 在第 {iteration} 次迭代收敛。")
                break
            log_likelihood_old = log_likelihood

    def predict_proba(self, X):
        N, D = X.shape
        gamma = np.zeros((N, self.K))
        for k in range(self.K):
            diff = X - self.mu[k]
            inv_Sigma = np.linalg.inv(self.Sigma[k] + 1e-6 * np.eye(D))
            det_Sigma = np.linalg.det(self.Sigma[k] + 1e-6 * np.eye(D))
            norm_const = 1.0 / (np.power(2 * np.pi, D / 2.0) * np.sqrt(det_Sigma) + 1e-15)
            exon = -0.5 * np.sum(np.dot(diff, inv_Sigma) * diff, axis=1)
            gamma[:, k] = self.pi[k] * norm_const * np.exp(exon)
        return gamma / np.sum(gamma, axis=1, keepdims=True)

# 验证 GMM 聚类
np.random.seed(42)
X1 = np.random.randn(100, 2) + np.array([3, 3])
X2 = np.random.randn(100, 2) + np.array([-3, -3])
X_data = np.vstack([X1, X2])

gmm = GaussianMixtureModel(n_components=2)
gmm.fit(X_data)
probs = gmm.predict_proba(X_data)
print(f"前 5 个样本属于集群 0 的概率: {probs[:5, 0]}")

```

