
# 20. 演员-评论家 (Actor-Critic, AC)

## 1. 核心原理

Actor-Critic 结合了基于策略（Policy-based）**和**基于价值（Value-based）两类方法的优势：

* **Actor（演员）**：负责选择动作，由策略网络 $\pi_\theta(a\vert{}s)$ 参数化。
* **Critic（评论家）**：负责评估动作的好坏，由价值网络 $V_\phi(s)$ 参数化。

Critic 估算状态价值函数 $V_\phi(s)$，并提供优势函数（Advantage Function）或 TD 误差来指导 Actor 更新。相比蒙特卡洛 REINFORCE 算法，Actor-Critic 能显著降低方差，提高收敛速度。

## 2. 算法与数学公式

### 2.1 时序差分误差 (TD Error) 作为优势函数

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### 2.2 Critic 损失函数

$$L_{\text{Critic}}(\phi) = \delta_t^2 = \left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2$$

### 2.3 Actor 策略梯度

$$\nabla_{\theta} J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi_\theta(a_t\vert{}s_t) \delta_t \right]$$


Actor 损失：


$$L_{\text{Actor}}(\theta) = -\log \pi_\theta(a_t\vert{}s_t) \cdot \delta_t.\text{detach()}$$

## 3. ASCII 结构框架图

```
                        +--------------------+
                        |    环境 (Env)      |
                        +---------+----------+
                                  | 状态 s
                                  v
                +-----------------+-----------------+
                |                                   |
                v                                   v
      +-------------------+               +-------------------+
      |   Actor (演员)    |               |   Critic (评论家) |
      |   pi_theta(a|s)   |               |   V_phi(s)        |
      +---------+---------+               +---------+---------+
                |                                   |
                v 采样动作 a                        v 计算价值 V(s)
       [执行动作得到 r, s']                [计算 TD-Error: delta]
                |                                   |
                +-----------------+-----------------+
                                  |
                                  v
                   +------------------------------+
                   |  delta = r + gamma*V(s')-V(s)|
                   +--------------+---------------+
                                  |
                   +--------------+--------------+
                   |                             |
                   v                             v
          更新 Actor 参数 theta          更新 Critic 参数 phi
         (梯度: log_prob * delta)        (损失: MSE(delta))

```

## 4. PyTorch 简易实现代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # 共享特征层或独立层
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        probs = self.actor(state)
        value = self.critic(state)
        return probs, value

class ActorCriticAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        self.ac_net = ActorCriticNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

    def train_step(self, state, action, reward, next_state, done):
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)

        probs, value = self.ac_net(state_t)
        _, next_value = self.ac_net(next_state_t)

        m = Categorical(probs)
        log_prob = m.log_prob(torch.tensor(action))

        # 计算 TD Error
        td_target = reward + (0 if done else self.gamma * next_value.item())
        td_error = td_target - value

        # Actor Loss & Critic Loss
        actor_loss = -log_prob * td_error.detach()
        critic_loss = td_error.pow(2)
        total_loss = actor_loss + critic_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    agent = ActorCriticAgent(state_dim=4, action_dim=2)
    agent.train_step(state=[0.1, 0.2, 0.3, 0.4], action=0, reward=1.0, next_state=[0.2, 0.3, 0.4, 0.5], done=False)
    print("Actor-Critic updated successfully.")

```

