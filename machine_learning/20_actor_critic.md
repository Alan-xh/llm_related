# 20. 演员-评论家 (Actor-Critic, AC)

## 1. 核心原理

Actor-Critic 结合了基于策略（Policy-based）**和**基于价值（Value-based）两类方法的优势：

* **Actor（演员）**：负责选择动作，由策略网络 $\pi_\theta(a\vert{}s)$ 参数化。
* $\pi$(派): 策略网络表示的概率分布
* $\theta$(西塔): 策略网络的参数
* a: 选取的动作 (Action)
* s: 当前环境的状态 (State)


* **Critic（评论家）**：负责评估动作的好坏，由价值网络 $V_\phi(s)$ 参数化。
* V: 状态价值函数 (Value Function)
* $\phi$(斐): 价值网络的参数
* s: 当前环境的状态 (State)



Critic 估算状态价值函数 $V_\phi(s)$，并提供优势函数（Advantage Function）或 TD 误差来指导 Actor 更新。相比蒙特卡洛 REINFORCE 算法，Actor-Critic 能显著降低方差，提高收敛速度。

## 2. 算法与数学公式

### 2.1 时序差分误差 (TD Error) 作为优势函数

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

* $\delta$(德尔塔): 时间 $t$ 的时序差分误差 (TD Error)
* t: 时间步/时刻指数
* $r_t$: 在时间步 $t$ 获得的即时奖励 (Reward)
* $\gamma$(伽马): 折扣因子 (Discount Factor)
* $V_\phi$: 参数为 $\phi$(斐) 的状态价值函数
* $s_{t+1}$: 时间步 $t+1$ 的下一个状态 (Next State)
* $s_t$: 时间步 $t$ 的当前状态 (Current State)

### 2.2 Critic 损失函数

$$L_{\text{Critic}}(\phi) = \delta_t^2 = \left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2$$

* $L_{\text{Critic}}$: 评论家 (Critic) 的损失函数 (Loss)
* $\phi$(斐): 价值网络的参数
* $\delta_t$: 时间步 $t$ 的 TD 误差 (TD Error)，具体定义见公式 2.1
* $r_t$: 时间步 $t$ 获得的即时奖励
* $\gamma$(伽马): 折扣因子
* $V_\phi(s_{t+1})$: 下一状态 $s_{t+1}$ 的估计价值
* $V_\phi(s_t)$: 当前状态 $s_t$ 的估计价值

### 2.3 Actor 策略梯度

$$\nabla_{\theta} J(\theta) = \mathbb{E} \left[ \nabla_{\theta} \log \pi_\theta(a_t\vert{}s_t) \delta_t \right]$$

* $\nabla_\theta$(纳布拉): 关于策略参数 $\theta$(西塔) 的偏导数/梯度算子
* $J(\theta)$: 策略的目标函数/期望回报 (Objective Function)
* $\mathbb{E}$: 期望值 (Expectation)
* $\pi_\theta(a_t\vert{}s_t)$: 在状态 $s_t$ 下采取动作 $a_t$ 的策略概率
* $\log$: 对数函数
* $a_t$: 时间步 $t$ 采样的动作
* $s_t$: 时间步 $t$ 的状态
* $\delta_t$: 时间步 $t$ 的 TD 误差，作为优势函数的替代标量

Actor 损失：

$$L_{\text{Actor}}(\theta) = -\log \pi_\theta(a_t\vert{}s_t) \cdot \delta_t.\text{detach()}$$

* $L_{\text{Actor}}$: 演员 (Actor) 的损失函数 (Loss)
* $\theta$(西塔): 策略网络的参数
* $\pi_\theta(a_t\vert{}s_t)$: 在状态 $s_t$ 下选择动作 $a_t$ 的概率
* $\delta_t$: 时间步 $t$ 的 TD 误差
* $\text{detach()}$: 阻断梯度反向传播的操作（将 TD 误差视为常数）

## 3. ASCII 结构框架图

```
                        +--------------------+
                        |     环境 (Env)      |
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