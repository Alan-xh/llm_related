
# 19. 策略梯度 (Policy Gradient / REINFORCE)

## 1. 核心原理

基于价值的方法（如 Q-Learning）通过估计 $Q(s, a)$ 间接获取策略，而策略梯度（Policy Gradient）直接将策略参数化为 $\pi_\theta(a\vert{}s)$，并使用梯度上升法直接优化期望累积回报 $J(\theta)$。

主要特点：

* 可以自然处理连续动作空间。
* 可以学习随机策略（Randomized Policy）。
* REINFORCE 算法是一种典型的基于蒙特卡洛（Monte Carlo）采样的策略梯度算法，即利用全轨迹的回报 $G_t$ 作为无偏估计更新参数。

## 2. 算法与数学公式

### 2.1 目标函数

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} [R(\tau)]$$

### 2.2 策略梯度定理 (Policy Gradient Theorem)

$$\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_{\theta} \log \pi_\theta(a_t\vert{}s_t) G_t \right]$$


其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_k$ 是从时刻 $t$ 开始的折扣累积回报（Return）。

### 2.3 参数更新公式 (梯度上升)

$$\theta \leftarrow \theta + \alpha \sum_{t=0}^{T} \nabla_{\theta} \log \pi_\theta(a_t\vert{}s_t) G_t$$

## 3. ASCII 流程框架图

```
+-------------------------------------------------------+
|  使用当前策略 pi_theta 采样完整轨迹 (Trajectory) tau    |
|  (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T)              |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  对每个时刻 t，计算累积折扣回报 G_t = sum(gamma^k * r) |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  计算策略损失 Loss = - sum( log pi_theta(a_t|s_t) * G_t )|
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  反向传播求解梯度并更新策略网络参数 theta               |
+-------------------------------------------------------+

```

## 4. PyTorch 简易实现代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.fc(x)

class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=1e-2, gamma=0.99):
        self.gamma = gamma
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
    def select_action(self, state):
        state_t = torch.FloatTensor(state)
        probs = self.policy_net(state_t)
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

    def update(self, log_probs, rewards):
        discounted_returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted_returns.insert(0, G)
        
        returns_t = torch.FloatTensor(discounted_returns)
        # 归一化 returns 降低方差
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        policy_loss = []
        for log_prob, g in zip(log_probs, returns_t):
            policy_loss.append(-log_prob * g)

        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    agent = REINFORCE(state_dim=4, action_dim=2)
    print("REINFORCE agent initialized.")

```

