# 第二章：随机过程基础、马尔可夫链与 MCMC 采样

## 1. 核心概念与数学表达

### 1.1 随机过程 (Stochastic Processes)

随机过程是定义在同一概率空间上的一族随机变量 $\{X(t), t \in T\}$。

* 若 index $T$ 为离散集（如 $\mathbb{N}$），称为**离散时间随机过程** $X_0, X_1, X_2, \dots$。
* **平稳随机过程 (Stationary Process)**：
* **严平稳**：任意有限维联合分布在时间平移下保持不变。
* **宽平稳 (Weakly/Wide-Sense Stationary, WSS)**：
1. 均值函数为常数：$\mathbb{E}[X(t)] = \mu$。
2. 自协方差函数只与时间差 $\tau$ 有关：$\gamma(t, t+\tau) = \mathbb{E}[(X(t)-\mu)(X(t+\tau)-\mu)] = R(\tau)$。





### 1.2 马尔可夫链 (Markov Chains)

满足无记忆性（马尔可夫性）的随机过程：


$$P(X_{n+1} = x_{n+1} \mid X_n = x_n, X_{n-1} = x_{n-1}, \dots, X_0 = x_0) = P(X_{n+1} = x_{n+1} \mid X_n = x_n)$$

* **转移概率矩阵 (Transition Matrix $P$)**：$P_{ij} = P(X_{n+1} = j \mid X_n = i)$，满足 $\sum_j P_{ij} = 1$。
* **平稳分布 (Stationary Distribution $\pi$)**：若概率向量 $\pi$ 满足：

$$\pi P = \pi, \quad \sum_i \pi_i = 1$$



则称 $\pi$ 为马尔可夫链的平稳分布。
* **细致平衡条件 (Detailed Balance Condition)**：若对所有状态 $i, j$，存在分布 $\pi$ 使得：

$$\pi_i P_{ij} = \pi_j P_{ji}$$



满足细致平衡条件的分布 $\pi$ 必然是该马尔可夫链的平稳分布（充分条件）。

---

## 2. 马尔可夫链蒙特卡洛采样 (MCMC)

### 2.1 Metropolis-Hastings (MH) 算法原理

为了从复杂的未归一化目标分布 $p(x) = \frac{\tilde{p}(x)}{Z}$ 中采样，引入提议分布 (Proposal Distribution) $q(x' \mid x)$。
为了满足细致平衡条件 $\pi(x) P(x \to x') = \pi(x') P(x' \to x)$，构造转移概率 $P(x \to x') = q(x' \mid x) \alpha(x, x')$，其中 $\alpha(x, x')$ 为**接受率 (Acceptance Rate)**：


$$\alpha(x, x') = \min \left( 1, \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)} \right) = \min \left( 1, \frac{\tilde{p}(x') q(x \mid x')}{\tilde{p}(x) q(x' \mid x)} \right)$$

### 2.2 吉布斯采样 (Gibbs Sampling)

吉布斯采样是 MH 算法的特例。在多维随机变量 $X = (X_1, X_2, \dots, X_d)$ 中，每次固定其他维度，依次从满条件分布 (Full Conditional Distribution) 中采样：


$$X_i^{(t+1)} \sim P(X_i \mid X_1^{(t+1)}, \dots, X_{i-1}^{(t+1)}, X_{i+1}^{(t)}, \dots, X_d^{(t)})$$


其接受率 $\alpha \equiv 1$（必定接受）。

---

## 3. AI/ML 经典应用案例：从复杂非标准双峰分布中采样的 Metropolis-Hastings 算法 Python 实现

### 3.1 目标分布定义

定义未归一化的双峰目标密度函数：


$$\tilde{p}(x) = \exp(-x^2) + 0.5 \exp(-(x - 4)^2)$$

### 3.2 Metropolis-Hastings 采样器完整代码

```python
import numpy as np

def target_density(x):
    # 未归一化的双峰高斯混合分布
    return np.exp(-x**2) + 0.5 * np.exp(-(x - 4.0)**2)

def metropolis_hastings_sampler(n_samples=10000, proposal_std=1.0):
    samples = np.zeros(n_samples)
    current_x = 0.0 # 初始状态
    accepted_count = 0
    
    for i in range(n_samples):
        # 1. 从对称提议分布 q(x' | x) = N(x, proposal_std^2) 生成候选样本
        proposed_x = np.random.normal(current_x, proposal_std)
        
        # 2. 计算接受率 alpha (因为 q 是对称的，q(x|x') = q(x'|x)，故可约简)
        p_current = target_density(current_x)
        p_proposed = target_density(proposed_x)
        
        alpha = min(1.0, p_proposed / p_current)
        
        # 3. 接受/拒绝判定
        if np.random.rand() < alpha:
            current_x = proposed_x
            accepted_count += 1
            
        samples[i] = current_x
        
    print(f"采样完成。接受率: {accepted_count / n_samples * 100:.2f}%")
    return samples

# 执行 MCMC 采样
np.random.seed(42)
mcmc_samples = metropolis_hastings_sampler(n_samples=20000, proposal_std=2.0)

# 丢弃 Burn-in 前预热阶段的样本
burn_in = 2000
final_samples = mcmc_samples[burn_in:]

print(f"采样均值: {np.mean(final_samples):.4f}")
print(f"采样方差: {np.var(final_samples):.4f}")

```

