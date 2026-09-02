# 18. 深度Q网络 (Deep Q-Network, DQN)

## 1. 核心原理

当状态空间是连续或极高维时（如图像输入），传统表格型 Q-learning 无法存储和求解。DQN 通过神经网络（Q网络 $Q(s, a; \theta)$）来近似状态-动作价值函数。

为了解决深度神经网络在强化学习中训练不稳定、数据关联性强的问题，DQN 引入了两个核心机制：

1. **经验回放（Experience Replay）**：将过渡样本 $(s, a, r, s', \text{done})$ 存储在 Replay Buffer 中，训练时随机采样打乱相关性，提高数据利用率。
2. **目标网络（Target Network）**：维持一个参数相对固定的目标 Q 网络 $Q(s, a; \theta^-)$，定期将当前 Q 网络的参数复制给目标网络，以稳定训练的 TD 目标值。

## 2. 算法与数学公式

### 2.1 目标 Q 值计算

$$y_i = r_i + \gamma (1 - \text{done}_i) \max_{a'} Q(s'_i, a'; \theta^-)$$

* $y_i$: 第 $i$ 个样本的目标 Q 值（Target Q-value）
* $r_i$: 第 $i$ 个样本获得的即时奖励（Reward）
* $\gamma$(伽马): 折扣因子（Discount Factor），控制未来奖励的现值权重
* $\text{done}_i$: 第 $i$ 个样本的终止状态标志位（终止状态为 1，非终止状态为 0）
* $a'$: 在下一状态 $s'_i$ 下可能采取的所有动作（Next Action）
* $s'_i$: 第 $i$ 个样本的下一个状态（Next State）
* $Q(s'_i, a'; \theta^-)$: 目标 Q 网络对下一状态-动作对估算的价值
* $\theta^-$(西塔负/目标西塔): 目标 Q 网络的网络参数

### 2.2 损失函数 (均方误差 MSE)

$$L(\theta) = \frac{1}{B} \sum_{i=1}^{B} \left( y_i - Q(s_i, a_i; \theta) \right)^2$$

* $L(\theta)$: 关于当前 Q 网络参数 $\theta$ 的损失函数（Loss Function）
* $\theta$(西塔): 当前 Q 网络的可训练权重参数
* $B$: 随机采样的批次大小（Batch Size）
* $i$: 批次中采样样本的索引
* $y_i$: 第 $i$ 个样本计算得到的目标 Q 值
* $s_i$: 第 $i$ 个样本的当前状态（Current State）
* $a_i$: 第 $i$ 个样本执行的动作（Current Action）
* $Q(s_i, a_i; \theta)$: 当前 Q 网络预测的在状态 $s_i$ 下采取动作 $a_i$ 的 Q 估算值

### 2.3 梯度更新

$$\nabla_{\theta} L(\theta) = -\frac{2}{B} \sum_{i=1}^{B} \left[ y_i - Q(s_i, a_i; \theta) \right] \nabla_{\theta} Q(s_i, a_i; \theta)$$

* $\nabla_{\theta}$(纳布拉/梯度): 损失函数对当前 Q 网络参数 $\theta$ 的梯度
* $L(\theta)$: 损失函数
* $B$: 随机采样的批次大小（Batch Size）
* $i$: 批次中采样样本的索引
* $y_i$: 第 $i$ 个样本的目标 Q 值
* $s_i$: 第 $i$ 个样本的当前状态
* $a_i$: 第 $i$ 个样本执行的动作
* $Q(s_i, a_i; \theta)$: 当前 Q 网络预测的 Q 估算值
* $\nabla_{\theta} Q(s_i, a_i; \theta)$: 当前 Q 网络输出对网络参数 $\theta$ 的偏导数/梯度

## 3. ASCII 结构框架图

```
 +---------------------------------------------------------+
 |                      环境 (Environment)                 |
 +-----------+---------------------------------+-----------+
             | s, r, done                      ^ a
             v                                 |
 +-----------+---------------------------------+-----------+
 |                     经验回放池 (Replay Buffer)          |
 |             存储样本 (s, a, r, s', done)                  |
 +---------------------------+-----------------------------+
                             | 随机小批量采样 (Batch)
                             v
           +-----------------+-----------------+
           |                                   |
           v                                   v
 +-------------------+               +-------------------+
 |  当前 Q 网络       |               |   目标 Q 网络     |
 |  Q(s, a; theta)   |               |   Q(s', a'; theta-)|
 +---------+---------+               +---------+---------+
           |                                   |
           v                                   v
  [ Q(s, a; theta) ]                [ r + gamma * max Q(s',a') ]
           \\                                 /
            \\-- MSE Loss = (Q - Target)^2 --/
                          |
                          v
                反向传播更新 theta
                定期更新 theta- <- theta


```

## 4. PyTorch 简易实现代码

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.action_dim = action_dim
        self.gamma = gamma
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=10000)

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.q_net(state_t).argmax(dim=1).item()

    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        # 当前 Q 值
        q_values = self.q_net(states_t).gather(1, actions_t)

        # 目标 Q 值
        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

if __name__ == "__main__":
    agent = DQNAgent(state_dim=4, action_dim=2)
    # 添加一个假数据
    agent.buffer.append(([0.1, 0.2, 0.3, 0.4], 1, 1.0, [0.2, 0.3, 0.4, 0.5], False))
    print("DQN agent initialized successfully.")


```